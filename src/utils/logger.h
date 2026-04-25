#pragma once
// ==============================================================================
// Logger - Logging utility for FCW System
// ==============================================================================

#include <string>
#include <fstream>
#include <mutex>
#include <memory>

namespace fcw {
namespace utils {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

class Logger {
public:
    static Logger& getInstance();

    void init(const std::string& logFile, LogLevel minLevel = LogLevel::INFO);
    void shutdown();

    void log(LogLevel level, const std::string& module, const std::string& message);

    void debug(const std::string& module, const std::string& msg);
    void info(const std::string& module, const std::string& msg);
    void warning(const std::string& module, const std::string& msg);
    void error(const std::string& module, const std::string& msg);
    void fatal(const std::string& module, const std::string& msg);

private:
    Logger() = default;
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::string levelToString(LogLevel level) const;
    std::string getCurrentTimestamp() const;

    std::ofstream logFile_;
    LogLevel minLevel_ = LogLevel::INFO;
    std::mutex mutex_;
    bool initialized_ = false;
};

// Convenience macros
#define LOG_DEBUG(module, msg)   fcw::utils::Logger::getInstance().debug(module, msg)
#define LOG_INFO(module, msg)    fcw::utils::Logger::getInstance().info(module, msg)
#define LOG_WARNING(module, msg) fcw::utils::Logger::getInstance().warning(module, msg)
#define LOG_ERROR(module, msg)   fcw::utils::Logger::getInstance().error(module, msg)
#define LOG_FATAL(module, msg)   fcw::utils::Logger::getInstance().fatal(module, msg)

} // namespace utils
} // namespace fcw
