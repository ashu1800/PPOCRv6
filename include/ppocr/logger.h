#pragma once

#include <string>

namespace ppocr::log {

void info(const std::string& message);
void warn(const std::string& message);
void error(const std::string& message);

} // namespace ppocr::log
