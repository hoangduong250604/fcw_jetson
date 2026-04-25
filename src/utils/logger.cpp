// ==============================================================================
// Logger Implementation
// ==============================================================================

#include "logger.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fcw {
namespace utils {

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

Logger::~Logger() {
    shutdown();
}

void Logger::init(const std::string& logFile, LogLevel minLevel) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) return;

    minLevel_ = minLevel;
    logFile_.open(logFile, std::ios::out | std::ios::app);
    if (!logFile_.is_open()) {
        std::cerr << "[Logger] Failed to open log file: " << logFile << std::endl;
    }
    initialized_ = true;

    // Log init message directly (can't call log() - would deadlock on mutex_)
    std::string msg = "[" + getCurrentTimestamp() + "] [INFO] [Logger] Logger initialized. Min level: " + levelToString(minLevel);
    std::cout << "\033[32m" << msg << "\033[0m" << std::endl;
    if (logFile_.is_open()) {
        logFile_ << msg << std::endl;
        logFile_.flush();
    }
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (logFile_.is_open()) {
        logFile_.close();
    }
    initialized_ = false;
}

void Logger::log(LogLevel level, const std::string& module, const std::string& message) {
    if (level < minLevel_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    std::string timestamp = getCurrentTimestamp();
    std::string levelStr = levelToString(level);

    std::ostringstream oss;
    oss << "[" << timestamp << "] [" << levelStr << "] [" << module << "] " << message;
    std::string logLine = oss.str();

    // Console output with colors
    switch (level) {
        case LogLevel::DEBUG:   std::cout << "\033[36m"; break;  // Cyan
        case LogLevel::INFO:    std::cout << "\033[32m"; break;  // Green
        case LogLevel::WARNING: std::cout << "\033[33m"; break;  // Yellow
        case LogLevel::ERROR:   std::cout << "\033[31m"; break;  // Red
        case LogLevel::FATAL:   std::cout << "\033[1;31m"; break; // Bold Red
    }
    std::cout << logLine << "\033[0m" << std::endl;

    // File output
    if (logFile_.is_open()) {
        logFile_ << logLine << std::endl;
        logFile_.flush();
    }
}

void Logger::debug(const std::string& module, const std::string& msg) {
    log(LogLevel::DEBUG, module, msg);
}

void Logger::info(const std::string& module, const std::string& msg) {
    log(LogLevel::INFO, module, msg);
}

void Logger::warning(const std::string& module, const std::string& msg) {
    log(LogLevel::WARNING, module, msg);
}

void Logger::error(const std::string& module, const std::string& msg) {
    log(LogLevel::ERROR, module, msg);
}

void Logger::fatal(const std::string& module, const std::string& msg) {
    log(LogLevel::FATAL, module, msg);
}

std::string Logger::levelToString(LogLevel level) const {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::FATAL:   return "FATAL";
        default:                return "?????";
    }
}

std::string Logger::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

} // namespace utils
} // namespace fcw
