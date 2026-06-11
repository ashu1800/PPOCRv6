#pragma once

#include "ppocr/ocr_engine.h"

#include <array>
#include <string>
#include <vector>

namespace ppocr {

struct CtcDecodeResult {
    std::string text;
    float confidence = 0.0F;
};

CtcDecodeResult ctc_decode(const std::vector<int>& indices, const std::vector<float>& probabilities, const std::vector<std::string>& keys);
std::array<Point, 4> order_box_points(const std::array<Point, 4>& points);
std::array<Point, 4> expand_text_box(const std::array<Point, 4>& ordered_box, float image_width, float image_height);
std::string merge_text_segments(const std::vector<std::string>& segments);
bool should_use_orientation_retry_result(
    const std::string& original_text,
    float original_confidence,
    const std::string& rotated_text,
    float rotated_confidence);
void sort_boxes_reading_order(std::vector<OcrItem>& items);

} // namespace ppocr
