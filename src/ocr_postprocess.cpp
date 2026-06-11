#include "ppocr/ocr_postprocess.h"

#include <algorithm>
#include <cmath>
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

float box_height(const OcrItem& item) {
    float min_y = item.box[0].y;
    float max_y = item.box[0].y;
    for (const auto& point : item.box) {
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }
    return std::max(1.0F, max_y - min_y);
}

float distance_between(const Point& lhs, const Point& rhs) {
    const float dx = lhs.x - rhs.x;
    const float dy = lhs.y - rhs.y;
    return std::sqrt(dx * dx + dy * dy);
}

Point unit_vector(const Point& from, const Point& to) {
    const float dx = to.x - from.x;
    const float dy = to.y - from.y;
    const float length = std::max(0.0001F, std::sqrt(dx * dx + dy * dy));
    return {dx / length, dy / length};
}

Point clamp_point(Point point, float image_width, float image_height) {
    const float max_x = std::max(0.0F, image_width - 1.0F);
    const float max_y = std::max(0.0F, image_height - 1.0F);
    point.x = std::clamp(point.x, 0.0F, max_x);
    point.y = std::clamp(point.y, 0.0F, max_y);
    return point;
}

bool is_utf8_boundary(const std::string& value, std::size_t index) {
    if (index == 0 || index == value.size()) {
        return true;
    }
    return (static_cast<unsigned char>(value[index]) & 0xC0U) != 0x80U;
}

std::size_t utf8_overlap_bytes(const std::string& lhs, const std::string& rhs) {
    const auto max_len = std::min(lhs.size(), rhs.size());
    for (std::size_t len = max_len; len > 0; --len) {
        const auto lhs_start = lhs.size() - len;
        if (!is_utf8_boundary(lhs, lhs_start) || !is_utf8_boundary(rhs, len)) {
            continue;
        }
        if (lhs.compare(lhs_start, len, rhs, 0, len) == 0) {
            return len;
        }
    }
    return 0;
}

std::size_t utf8_character_count(const std::string& value) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (is_utf8_boundary(value, i)) {
            ++count;
        }
    }
    return count;
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

std::array<Point, 4> order_box_points(const std::array<Point, 4>& points) {
    Point center{};
    for (const auto& point : points) {
        center.x += point.x;
        center.y += point.y;
    }
    center.x /= static_cast<float>(points.size());
    center.y /= static_cast<float>(points.size());

    std::array<Point, 4> ordered = points;
    std::sort(ordered.begin(), ordered.end(), [&center](const Point& lhs, const Point& rhs) {
        const float lhs_angle = std::atan2(lhs.y - center.y, lhs.x - center.x);
        const float rhs_angle = std::atan2(rhs.y - center.y, rhs.x - center.x);
        return lhs_angle < rhs_angle;
    });

    auto top_left = ordered.begin();
    for (auto it = ordered.begin(); it != ordered.end(); ++it) {
        if (it->y < top_left->y || (std::abs(it->y - top_left->y) <= 0.001F && it->x < top_left->x)) {
            top_left = it;
        }
    }
    std::rotate(ordered.begin(), top_left, ordered.end());

    const Point edge_a{ordered[1].x - ordered[0].x, ordered[1].y - ordered[0].y};
    const Point edge_b{ordered[2].x - ordered[1].x, ordered[2].y - ordered[1].y};
    const float cross = edge_a.x * edge_b.y - edge_a.y * edge_b.x;
    if (cross < 0.0F) {
        std::swap(ordered[1], ordered[3]);
    }

    return ordered;
}

std::array<Point, 4> expand_text_box(const std::array<Point, 4>& ordered_box, float image_width, float image_height) {
    const float top_width = distance_between(ordered_box[0], ordered_box[1]);
    const float bottom_width = distance_between(ordered_box[3], ordered_box[2]);
    const float left_height = distance_between(ordered_box[0], ordered_box[3]);
    const float right_height = distance_between(ordered_box[1], ordered_box[2]);
    const float width = std::max(top_width, bottom_width);
    const float height = std::max(left_height, right_height);
    const float pad_x = std::clamp(width * 0.03F, 2.0F, 24.0F);
    const float pad_y = std::clamp(height * 0.10F, 2.0F, 12.0F);

    const Point top = unit_vector(ordered_box[0], ordered_box[1]);
    const Point bottom = unit_vector(ordered_box[3], ordered_box[2]);
    const Point left = unit_vector(ordered_box[0], ordered_box[3]);
    const Point right = unit_vector(ordered_box[1], ordered_box[2]);

    std::array<Point, 4> expanded{
        Point{ordered_box[0].x - top.x * pad_x - left.x * pad_y, ordered_box[0].y - top.y * pad_x - left.y * pad_y},
        Point{ordered_box[1].x + top.x * pad_x - right.x * pad_y, ordered_box[1].y + top.y * pad_x - right.y * pad_y},
        Point{ordered_box[2].x + bottom.x * pad_x + right.x * pad_y, ordered_box[2].y + bottom.y * pad_x + right.y * pad_y},
        Point{ordered_box[3].x - bottom.x * pad_x + left.x * pad_y, ordered_box[3].y - bottom.y * pad_x + left.y * pad_y},
    };

    for (auto& point : expanded) {
        point = clamp_point(point, image_width, image_height);
    }
    return expanded;
}

std::string merge_text_segments(const std::vector<std::string>& segments) {
    std::string merged;
    for (const auto& segment : segments) {
        if (segment.empty()) {
            continue;
        }
        if (merged.empty()) {
            merged = segment;
            continue;
        }
        const auto overlap = utf8_overlap_bytes(merged, segment);
        merged += segment.substr(overlap);
    }
    return merged;
}

bool should_use_orientation_retry_result(
    const std::string& original_text,
    float original_confidence,
    const std::string& rotated_text,
    float rotated_confidence) {
    if (rotated_text.empty()) {
        return false;
    }
    if (rotated_confidence < original_confidence + 0.05F) {
        return false;
    }

    const auto original_length = utf8_character_count(original_text);
    if (original_length == 0) {
        return true;
    }
    const auto rotated_length = utf8_character_count(rotated_text);
    return static_cast<float>(rotated_length) >= static_cast<float>(original_length) * 0.70F;
}

void sort_boxes_reading_order(std::vector<OcrItem>& items) {
    if (items.size() < 2) {
        return;
    }

    std::vector<float> heights;
    heights.reserve(items.size());
    for (const auto& item : items) {
        heights.push_back(box_height(item));
    }
    std::nth_element(heights.begin(), heights.begin() + static_cast<std::ptrdiff_t>(heights.size() / 2), heights.end());
    const float median_height = heights[heights.size() / 2];
    const float line_threshold = std::max(12.0F, median_height * 0.65F);

    std::sort(items.begin(), items.end(), [](const OcrItem& lhs, const OcrItem& rhs) {
        const float lhs_y = average_y(lhs);
        const float rhs_y = average_y(rhs);
        if (std::abs(lhs_y - rhs_y) > 0.001F) {
            return lhs_y < rhs_y;
        }
        return min_x(lhs) < min_x(rhs);
    });

    struct Line {
        float center_y = 0.0F;
        std::vector<OcrItem> line_items;
    };
    std::vector<Line> lines;
    for (auto& item : items) {
        const float item_y = average_y(item);
        if (lines.empty() || std::abs(item_y - lines.back().center_y) > line_threshold) {
            lines.push_back({item_y, {std::move(item)}});
        } else {
            auto& line = lines.back();
            line.center_y = (line.center_y * static_cast<float>(line.line_items.size()) + item_y) /
                            static_cast<float>(line.line_items.size() + 1);
            line.line_items.push_back(std::move(item));
        }
    }

    items.clear();
    for (auto& line : lines) {
        std::sort(line.line_items.begin(), line.line_items.end(), [](const OcrItem& lhs, const OcrItem& rhs) {
            return min_x(lhs) < min_x(rhs);
        });
        for (auto& item : line.line_items) {
            items.push_back(std::move(item));
        }
    }
}

} // namespace ppocr
