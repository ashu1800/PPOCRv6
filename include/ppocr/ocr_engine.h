#pragma once

#include "ppocr/config.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ppocr {

struct Point {
    float x = 0.0F;
    float y = 0.0F;
};

struct OcrItem {
    std::string text;
    float confidence = 0.0F;
    std::array<Point, 4> box{};
};

struct OcrResult {
    std::string full_text;
    std::vector<OcrItem> items;
};

struct RuntimeInfo {
    std::string inference_mode = "CPU";
    bool gpu_enabled = false;
    std::string gpu_device_name;
    std::string fallback_reason;
};

class OcrEngine {
public:
    virtual ~OcrEngine() = default;
    virtual OcrResult recognize(const std::vector<std::uint8_t>& image_bytes) = 0;
    virtual const RuntimeInfo& runtime_info() const = 0;
};

std::shared_ptr<OcrEngine> create_ocr_engine(const Config& config, const std::filesystem::path& runtime_dir);

} // namespace ppocr
