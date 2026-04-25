#pragma once
// ==============================================================================
// Timer - Performance measurement utility
// ==============================================================================

#include <string>
#include <chrono>
#include <unordered_map>

namespace fcw {
namespace utils {

class Timer {
public:
    Timer();

    /** Start timing a named section */
    void start(const std::string& name);

    /** Stop timing and return elapsed ms */
    double stop(const std::string& name);

    /** Get last recorded time for a section (ms) */
    double getLastTime(const std::string& name) const;

    /** Get average time for a section (ms) */
    double getAverageTime(const std::string& name) const;

    /** Get current FPS based on frame processing time */
    double getFPS() const;

    /** Record a full frame tick (call once per frame) */
    void frameTick();

    /** Print all recorded timings to console */
    void printSummary() const;

    /** Reset all timings */
    void reset();

private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    struct TimingRecord {
        TimePoint startTime;
        double lastMs = 0.0;
        double totalMs = 0.0;
        int count = 0;
    };

    std::unordered_map<std::string, TimingRecord> records_;

    // FPS tracking
    TimePoint lastFrameTime_;
    double frameDeltaMs_ = 0.0;
    double fpsSmoothed_ = 0.0;
    bool firstFrame_ = true;
};

/**
 * RAII scoped timer - measures time for a scope automatically.
 * Usage: { ScopedTimer st(timer, "detection"); ... detection code ... }
 */
class ScopedTimer {
public:
    ScopedTimer(Timer& timer, const std::string& name);
    ~ScopedTimer();

private:
    Timer& timer_;
    std::string name_;
};

} // namespace utils
} // namespace fcw
