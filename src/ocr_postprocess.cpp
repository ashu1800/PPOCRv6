#include "ppocr/ocr_postprocess.h"

#include <algorithm>
#include <numeric>

namespace ppocr {
namespace {

float average_y(const OcrItem& item) {
    float total = 0.0F;
    for (const auto& point : item.box) {
        total += point.y;
    }
    return total / static_cast<float>(item.box.size());
}

float min_x(const OcrItem& item) {
    float value = item.box[0].x;
    for (const auto& point : item.box) {
        value = std::min(value, point.x);
    }
    return value;
}

} // namespace

CtcDecodeResult ctc_decode(const std::vector<int>& indices, const std::vector<float>& probabilities, const std::vector<std::string>& keys) {
    CtcDecodeResult result;
    float confidence_sum = 0.0F;
    int confidence_count = 0;
    int previous = -1;

    for (std::size_t i = 0; i < indices.size(); ++i) {
        const int index = indices[i];
        const float probability = i < probabilities.size() ? probabilities[i] : 0.0F;

        if (index > 0 && index != previous) {
            const auto key_index = static_cast<std::size_t>(index - 1);
            if (key_index < keys.size()) {
                result.text += keys[key_index];
                confidence_sum += probability;
                ++confidence_count;
            }
        }
        previous = index;
    }

    result.confidence = confidence_count == 0 ? 0.0F : confidence_sum / static_cast<float>(confidence_count);
    return result;
}

void sort_boxes_reading_order(std::vector<OcrItem>& items) {
    std::sort(items.begin(), items.end(), [](const OcrItem& lhs, const OcrItem& rhs) {
        const float lhs_y = average_y(lhs);
        const float rhs_y = average_y(rhs);
        if (std::abs(lhs_y - rhs_y) > 12.0F) {
            return lhs_y < rhs_y;
        }
        return min_x(lhs) < min_x(rhs);
    });
}

} // namespace ppocr
