#include "ppocr/logger.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace ppocr::log {
namespace {

std::mutex& log_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return out.str();
}

void write(const char* level, const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex());
    std::cout << "[" << timestamp() << "] [" << level << "] " << message << std::endl;
}

} // namespace

void info(const std::string& message) {
    write("INFO", message);
}

void warn(const std::string& message) {
    write("WARN", message);
}

void error(const std::string& message) {
    write("ERROR", message);
}

} // namespace ppocr::log
