// ==============================================================================
// Timer Implementation
// ==============================================================================

#include "timer.h"
#include "logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>

namespace fcw {
namespace utils {

Timer::Timer() : lastFrameTime_(Clock::now()) {}

void Timer::start(const std::string& name) {
    records_[name].startTime = Clock::now();
}

double Timer::stop(const std::string& name) {
    auto endTime = Clock::now();
    auto it = records_.find(name);
    if (it == records_.end()) return 0.0;

    auto& rec = it->second;
    double elapsed = std::chrono::duration<double, std::milli>(
        endTime - rec.startTime).count();

    rec.lastMs = elapsed;
    rec.totalMs += elapsed;
    rec.count++;

    return elapsed;
}

double Timer::getLastTime(const std::string& name) const {
    auto it = records_.find(name);
    return (it != records_.end()) ? it->second.lastMs : 0.0;
}

double Timer::getAverageTime(const std::string& name) const {
    auto it = records_.find(name);
    if (it == records_.end() || it->second.count == 0) return 0.0;
    return it->second.totalMs / it->second.count;
}

double Timer::getFPS() const {
    return fpsSmoothed_;
}

void Timer::frameTick() {
    auto now = Clock::now();
    if (firstFrame_) {
        firstFrame_ = false;
        lastFrameTime_ = now;
        return;
    }

    frameDeltaMs_ = std::chrono::duration<double, std::milli>(
        now - lastFrameTime_).count();
    lastFrameTime_ = now;

    double instantFPS = (frameDeltaMs_ > 0.0) ? (1000.0 / frameDeltaMs_) : 0.0;
    // EMA smoothing for FPS
    constexpr double alpha = 0.1;
    fpsSmoothed_ = (fpsSmoothed_ == 0.0) ? instantFPS
                                          : alpha * instantFPS + (1.0 - alpha) * fpsSmoothed_;
}

void Timer::printSummary() const {
    std::ostringstream oss;
    oss << "\n===== Timer Summary =====\n";
    oss << std::left << std::setw(25) << "Section"
        << std::right << std::setw(10) << "Last(ms)"
        << std::setw(10) << "Avg(ms)"
        << std::setw(8) << "Count" << "\n";
    oss << std::string(53, '-') << "\n";

    for (const auto& [name, rec] : records_) {
        double avg = (rec.count > 0) ? (rec.totalMs / rec.count) : 0.0;
        oss << std::left << std::setw(25) << name
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(10) << rec.lastMs
            << std::setw(10) << avg
            << std::setw(8) << rec.count << "\n";
    }
    oss << "FPS (smoothed): " << std::fixed << std::setprecision(1) << fpsSmoothed_ << "\n";
    oss << "=========================\n";

    LOG_INFO("Timer", oss.str());
}

void Timer::reset() {
    records_.clear();
    firstFrame_ = true;
    fpsSmoothed_ = 0.0;
}

// -- ScopedTimer --
ScopedTimer::ScopedTimer(Timer& timer, const std::string& name)
    : timer_(timer), name_(name) {
    timer_.start(name_);
}

ScopedTimer::~ScopedTimer() {
    timer_.stop(name_);
}

} // namespace utils
} // namespace fcw
