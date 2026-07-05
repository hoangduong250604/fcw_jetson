// ==============================================================================
// Warning System Implementation - Audio + Thread + Cooldown
// ==============================================================================
// Cross-platform audio:
//   Windows: Beep() for instant tonal alerts (no .wav files needed)
//   Linux/Jetson: aplay for .wav playback (async, non-blocking)
//
// Warning thread architecture:
//   Detection thread calls pushWarning(risk) → sets latestRisk_ + notifies CV
//   Warning thread wakes up → checks cooldown → plays audio if allowed
//   This prevents audio from blocking detection/tracking pipeline.
// ==============================================================================

#include "warning_system.h"
#include "logger.h"

#include <chrono>
#include <cstdlib>

#ifdef _WIN32
  #include <windows.h>
#endif

namespace fcw {

WarningSystem::WarningSystem() {}

WarningSystem::WarningSystem(const WarningConfig& config) : config_(config) {}

WarningSystem::~WarningSystem() {
    stopThread();
}

// ==============================================================================
// Push Warning (called from detection/processing thread)
// ==============================================================================
void WarningSystem::pushWarning(const RiskAssessment& risk) {
    currentLevel_.store(risk.level);

    // Always notify thread (even for SAFE, so it can stop audio after linger)
    {
        std::lock_guard<std::mutex> lock(warningMutex_);
        latestRisk_ = risk;
        hasNewRisk_ = true;
    }
    warningCv_.notify_one();

    if (risk.level == RiskLevel::SAFE) return;
    if (!config_.audioEnabled || muted_.load()) return;

    // If thread not running, play inline (fallback for single-threaded Pipeline)
    if (!threadRunning_.load()) {
        int64_t now = nowMs();
        int cooldown = getCooldownMs(risk.level);
        int64_t* lastTime = nullptr;

        switch (risk.level) {
            case RiskLevel::CRITICAL: lastTime = &lastCriticalTimeMs_; break;
            case RiskLevel::DANGER:   lastTime = &lastDangerTimeMs_; break;
            case RiskLevel::CAUTION:  lastTime = &lastCautionTimeMs_; break;
            default: return;
        }

        if (now - *lastTime >= cooldown) {
            *lastTime = now;
            playAudio(risk.level);

            LOG_WARNING("Warning", "ALERT [" + riskLevelToString(risk.level) + "] "
                        "Track:" + std::to_string(risk.trackId) +
                        " TTC:" + std::to_string(risk.ttc) + "s"
                        " Dist:" + std::to_string(risk.distance) + "m");
        }
    }
}

// ==============================================================================
// Warning Thread
// ==============================================================================
void WarningSystem::startThread() {
    if (threadRunning_.load()) return;

    threadShouldStop_.store(false);
    threadRunning_.store(true);
    warningThread_ = std::thread(&WarningSystem::warningThreadFunc, this);

    LOG_INFO("Warning", "Warning thread started");
}

void WarningSystem::stopThread() {
    if (!threadRunning_.load()) return;

    threadShouldStop_.store(true);
    warningCv_.notify_all();

    if (warningThread_.joinable()) {
        warningThread_.join();
    }
    threadRunning_.store(false);
    LOG_INFO("Warning", "Warning thread stopped");
}

void WarningSystem::warningThreadFunc() {
    LOG_INFO("Warning", "Warning monitor thread running");

    while (!threadShouldStop_.load()) {
        RiskAssessment risk;

        // Wait for new risk data (with timeout to check stop flag)
        {
            std::unique_lock<std::mutex> lock(warningMutex_);
            warningCv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
                return hasNewRisk_ || threadShouldStop_.load();
            });

            if (threadShouldStop_.load()) break;
            if (!hasNewRisk_) {
                // No new data — check if audio linger expired
                if (audioPlaying_) {
                    int64_t now = nowMs();
                    if (now - lastActiveTimeMs_ > kAudioLingerMs) {
#ifndef _WIN32
                        // Kill any lingering aplay processes
                        system("killall -q aplay 2>/dev/null");
#endif
                        audioPlaying_ = false;
                    }
                }
                continue;
            }

            risk = latestRisk_;
            hasNewRisk_ = false;
        }

        if (risk.level == RiskLevel::SAFE) {
            // Don't stop immediately — let audio linger
            continue;
        }

        // Active risk — update last active time
        lastActiveTimeMs_ = nowMs();
        if (muted_.load()) continue;

        // Cooldown check (per risk level)
        int64_t now = nowMs();
        int cooldown = getCooldownMs(risk.level);
        int64_t* lastTime = nullptr;

        switch (risk.level) {
            case RiskLevel::CRITICAL: lastTime = &lastCriticalTimeMs_; break;
            case RiskLevel::DANGER:   lastTime = &lastDangerTimeMs_; break;
            case RiskLevel::CAUTION:  lastTime = &lastCautionTimeMs_; break;
            default: continue;
        }

        if (now - *lastTime < cooldown) continue;
        *lastTime = now;

        // Play audio
        playAudio(risk.level);
        audioPlaying_ = true;

        LOG_WARNING("Warning", "ALERT [" + riskLevelToString(risk.level) + "] "
                    "Track:" + std::to_string(risk.trackId) +
                    " TTC:" + std::to_string(risk.ttc) + "s"
                    " Dist:" + std::to_string(risk.distance) + "m");
    }

    LOG_INFO("Warning", "Warning monitor thread exiting");
}

// ==============================================================================
// Audio Playback (Cross-Platform)
// ==============================================================================
void WarningSystem::playAudio(RiskLevel level) {
    // CAUTION intentionally plays no sound: it fires frequently in normal
    // traffic (the whole point of a "monitor, not yet dangerous" level), and
    // production ADAS/FCW products generally reserve audible alerts for
    // DANGER/CRITICAL to avoid alert fatigue from constant CAUTION chirps.
    if (level == RiskLevel::CAUTION) return;

#ifdef _WIN32
    // Windows: Use Beep() for instant tonal alerts
    // Different frequencies/durations per risk level
    int freq = 0, dur = 0;
    switch (level) {
        case RiskLevel::CRITICAL:
            freq = config_.criticalFreqHz;
            dur = config_.criticalDurationMs;
            break;
        case RiskLevel::DANGER:
            freq = config_.dangerFreqHz;
            dur = config_.dangerDurationMs;
            break;
        default:
            return;
    }

    // Beep is synchronous but short enough (~100-200ms) on warning thread
    Beep(static_cast<DWORD>(freq), static_cast<DWORD>(dur));

    // Double beep for CRITICAL
    if (level == RiskLevel::CRITICAL) {
        Beep(static_cast<DWORD>(freq + 200), static_cast<DWORD>(dur));
    }

#else
    // Linux / Jetson Nano: Use aplay for .wav playback (non-blocking with &)
    std::string soundFile;
    switch (level) {
        case RiskLevel::CRITICAL: soundFile = config_.criticalSound; break;
        case RiskLevel::DANGER:   soundFile = config_.dangerSound; break;
        default: return;
    }

    std::string fullPath = config_.soundsDir + soundFile;
    // Run aplay async (non-blocking). stderr redirected to /dev/null:
    // firing CRITICAL/DANGER alerts back-to-back (cooldowns as low as 400ms)
    // spawns overlapping aplay processes; when one exits while another is
    // mid-write to the shared ALSA device, the OS can interrupt the other's
    // write() with EINTR, which aplay reports as "pcm_write: write error:
    // Interrupted system call" on stderr. Harmless (that one alert's sound
    // is just cut short) but was showing up unfiltered in the console/log.
    std::string cmd = "aplay -q \"" + fullPath + "\" 2>/dev/null &";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        // Fallback: try paplay (PulseAudio)
        cmd = "paplay \"" + fullPath + "\" 2>/dev/null &";
        system(cmd.c_str());
    }
#endif
}

// ==============================================================================
// Helpers
// ==============================================================================
int WarningSystem::getCooldownMs(RiskLevel level) const {
    switch (level) {
        case RiskLevel::CRITICAL: return config_.criticalCooldownMs;
        case RiskLevel::DANGER:   return config_.dangerCooldownMs;
        case RiskLevel::CAUTION:  return config_.cautionCooldownMs;
        default: return 5000;
    }
}

int64_t WarningSystem::nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void WarningSystem::setConfig(const WarningConfig& config) {
    config_ = config;
}

} // namespace fcw
