#pragma once
// ==============================================================================
// Warning System - Alert driver of collision risk
// ==============================================================================
// Features:
//   1. Audio warning with cooldown (no spam)
//   2. Separate warning thread (non-blocking to detection pipeline)
//   3. Cross-platform: Windows (Beep), Linux/Jetson (aplay .wav)
//   4. Escalation: different sound patterns per risk level
// ==============================================================================

#include "risk_state.h"
#include <string>
#include <cstdint>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>

namespace fcw {

struct WarningConfig {
    bool audioEnabled = true;
    bool visualEnabled = true;

    // Cooldown intervals (ms) — prevent spam per risk level
    int criticalCooldownMs = 400;     // CRITICAL: beep every 0.4s (urgent)
    int dangerCooldownMs = 800;       // DANGER: every 0.8s
    int cautionCooldownMs = 2000;     // CAUTION: every 2s

    // Audio files (for Linux/Jetson .wav playback)
    std::string soundsDir = "./sounds/";
    std::string criticalSound = "critical_warning.wav";
    std::string dangerSound = "danger_warning.wav";
    std::string cautionSound = "caution_warning.wav";

    // Windows Beep parameters (frequency Hz, duration ms)
    int criticalFreqHz = 1800;
    int criticalDurationMs = 150;
    int dangerFreqHz = 1200;
    int dangerDurationMs = 200;
    int cautionFreqHz = 800;
    int cautionDurationMs = 100;
};

/**
 * Warning system with dedicated thread for non-blocking audio alerts.
 * 
 * Architecture:
 *   Pipeline/Detection thread → pushWarning(risk) → Warning thread plays audio
 *   
 * The warning thread runs independently, checking for new risk data
 * and playing audio with proper cooldown intervals.
 */
class WarningSystem {
public:
    WarningSystem();
    explicit WarningSystem(const WarningConfig& config);
    ~WarningSystem();

    /**
     * Push new risk assessment (called from detection/processing thread).
     * Non-blocking: just updates the latest risk level.
     */
    void pushWarning(const RiskAssessment& risk);

    /**
     * Legacy trigger (for backward compatibility with Pipeline::processFrame).
     * Same as pushWarning.
     */
    void trigger(const RiskAssessment& risk) { pushWarning(risk); }

    /** Start the warning thread */
    void startThread();

    /** Stop the warning thread */
    void stopThread();

    /** Check if warning thread is running */
    bool isThreadRunning() const { return threadRunning_.load(); }

    /** Set config */
    void setConfig(const WarningConfig& config);

    /** Get current active risk level */
    RiskLevel getCurrentLevel() const { return currentLevel_.load(); }

    /** Mute / unmute */
    void setMuted(bool muted) { muted_.store(muted); }
    bool isMuted() const { return muted_.load(); }

private:
    /** Warning thread entry point */
    void warningThreadFunc();

    /** Play audio alert for given risk level */
    void playAudio(RiskLevel level);

    /** Get cooldown for a risk level */
    int getCooldownMs(RiskLevel level) const;

    /** Get time in ms since epoch */
    static int64_t nowMs();

    WarningConfig config_;
    std::atomic<RiskLevel> currentLevel_{RiskLevel::SAFE};
    std::atomic<bool> muted_{false};

    // Per-level cooldown tracking
    int64_t lastCriticalTimeMs_ = 0;
    int64_t lastDangerTimeMs_ = 0;
    int64_t lastCautionTimeMs_ = 0;

    // Warning thread
    std::thread warningThread_;
    std::atomic<bool> threadRunning_{false};
    std::atomic<bool> threadShouldStop_{false};
    std::mutex warningMutex_;
    std::condition_variable warningCv_;
    RiskAssessment latestRisk_;
    bool hasNewRisk_ = false;
};

} // namespace fcw
