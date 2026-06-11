#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ppocr {

std::vector<std::uint8_t> decode_base64(const std::string& input);

} // namespace ppocr
