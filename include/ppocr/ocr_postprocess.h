#pragma once

#include "ppocr/ocr_engine.h"

#include <string>
#include <vector>

namespace ppocr {

struct CtcDecodeResult {
    std::string text;
    float confidence = 0.0F;
};

CtcDecodeResult ctc_decode(const std::vector<int>& indices, const std::vector<float>& probabilities, const std::vector<std::string>& keys);
void sort_boxes_reading_order(std::vector<OcrItem>& items);

} // namespace ppocr
