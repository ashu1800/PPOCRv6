#include "ppocr/base64.h"
#include "ppocr/config.h"
#include "ppocr/metrics.h"
#include "ppocr/ocr_postprocess.h"
#include "ppocr/request_queue.h"
#include "ppocr/usage.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string& message = "check failed") {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void run_test(const char* name, void (*test)()) {
    try {
        test();
    } catch (const std::exception& ex) {
        std::cerr << name << " failed: " << ex.what() << "\n";
        throw;
    }
}

void test_default_config_is_created() {
    const auto dir = std::filesystem::temp_directory_path() / "ppocrv6_config_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const auto path = dir / "config.json";
    const auto config = ppocr::Config::load_or_create(path);

    check(std::filesystem::exists(path));
    check(config.model_level == "small");
    check(config.listen_host == "0.0.0.0");
    check(config.port == 8080);
    check(config.prefer_gpu);
    check(!config.api_key.empty());
    check(config.max_concurrent_requests > 0);
    check(config.queue_size >= config.max_concurrent_requests);
    check(config.max_request_body_bytes == 16 * 1024 * 1024);
    check(config.rec_max_width == 960);
    check(config.rec_direct_max_width == 4096);
    check(config.enable_orientation_retry);

    std::filesystem::remove_all(dir);
}

void test_config_reads_and_validates_rec_options() {
    const auto dir = std::filesystem::temp_directory_path() / "ppocrv6_rec_config_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto path = dir / "config.json";

    std::ofstream out(path);
    out << R"({
        "model_level": "small",
        "listen_host": "0.0.0.0",
        "port": 8080,
        "api_key": "abc",
        "prefer_gpu": true,
        "gpu_device": 0,
        "max_concurrent_requests": 2,
        "queue_size": 4,
        "rec_max_width": 1280,
        "rec_direct_max_width": 6144,
        "enable_orientation_retry": false
    })";
    out.close();

    const auto config = ppocr::Config::load_or_create(path);

    check(config.rec_max_width == 1280);
    check(config.rec_direct_max_width == 6144);
    check(!config.enable_orientation_retry);
    std::filesystem::remove_all(dir);
}

void test_config_rejects_invalid_rec_max_width() {
    ppocr::Config config;
    config.api_key = "abc";
    config.rec_max_width = 4097;

    bool threw = false;
    try {
        config.validate();
    } catch (const std::exception&) {
        threw = true;
    }

    check(threw);
}

void test_config_rejects_invalid_model_level() {
    const auto dir = std::filesystem::temp_directory_path() / "ppocrv6_invalid_config_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto path = dir / "config.json";

    std::ofstream out(path);
    out << R"({
        "model_level": "large",
        "listen_host": "0.0.0.0",
        "port": 8080,
        "api_key": "abc",
        "prefer_gpu": true,
        "gpu_device": 0,
        "max_concurrent_requests": 2,
        "queue_size": 4
    })";
    out.close();

    bool threw = false;
    try {
        (void)ppocr::Config::load_or_create(path);
    } catch (const std::exception&) {
        threw = true;
    }

    check(threw);
    std::filesystem::remove_all(dir);
}

void test_base64_decodes_plain_and_data_url() {
    const auto plain = ppocr::decode_base64("SGVsbG8=");
    check(std::string(plain.begin(), plain.end()) == "Hello");

    const auto data_url = ppocr::decode_base64("data:image/png;base64,SGVsbG8=");
    check(std::string(data_url.begin(), data_url.end()) == "Hello");
}

void test_base64_rejects_invalid_input() {
    bool threw = false;
    try {
        (void)ppocr::decode_base64("%%%%");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw);
}

void test_request_queue_rejects_when_full() {
    ppocr::RequestQueue<int> queue(1);
    check(queue.try_push(1));
    check(!queue.try_push(2));

    int value = 0;
    check(queue.wait_pop(value));
    check(value == 1);
}

void test_ctc_decode_skips_blank_and_repeated_indices() {
    const std::vector<std::string> keys{"A", "B", "C"};
    const auto decoded = ppocr::ctc_decode({1, 1, 0, 2, 2, 3}, {0.9F, 0.8F, 0.4F, 0.7F, 0.6F, 0.5F}, keys);

    check(decoded.text == "ABC");
    check(decoded.confidence > 0.0F);
}

void test_boxes_sort_in_reading_order() {
    ppocr::OcrItem bottom;
    bottom.text = "bottom";
    bottom.box[0] = {10.0F, 50.0F};
    bottom.box[1] = {20.0F, 50.0F};
    bottom.box[2] = {20.0F, 60.0F};
    bottom.box[3] = {10.0F, 60.0F};

    ppocr::OcrItem top_right;
    top_right.text = "top_right";
    top_right.box[0] = {40.0F, 10.0F};
    top_right.box[1] = {50.0F, 10.0F};
    top_right.box[2] = {50.0F, 20.0F};
    top_right.box[3] = {40.0F, 20.0F};

    ppocr::OcrItem top_left;
    top_left.text = "top_left";
    top_left.box[0] = {5.0F, 12.0F};
    top_left.box[1] = {15.0F, 12.0F};
    top_left.box[2] = {15.0F, 22.0F};
    top_left.box[3] = {5.0F, 22.0F};

    std::vector<ppocr::OcrItem> items{bottom, top_right, top_left};
    ppocr::sort_boxes_reading_order(items);

    check(items[0].text == "top_left");
    check(items[1].text == "top_right");
    check(items[2].text == "bottom");
}

void test_boxes_sort_uses_dynamic_line_height_for_large_images() {
    ppocr::OcrItem top_right;
    top_right.text = "top_right";
    top_right.box[0] = {300.0F, 1000.0F};
    top_right.box[1] = {500.0F, 1000.0F};
    top_right.box[2] = {500.0F, 1060.0F};
    top_right.box[3] = {300.0F, 1060.0F};

    ppocr::OcrItem bottom;
    bottom.text = "bottom";
    bottom.box[0] = {10.0F, 1095.0F};
    bottom.box[1] = {220.0F, 1095.0F};
    bottom.box[2] = {220.0F, 1155.0F};
    bottom.box[3] = {10.0F, 1155.0F};

    ppocr::OcrItem top_left;
    top_left.text = "top_left";
    top_left.box[0] = {20.0F, 1008.0F};
    top_left.box[1] = {220.0F, 1008.0F};
    top_left.box[2] = {220.0F, 1068.0F};
    top_left.box[3] = {20.0F, 1068.0F};

    std::vector<ppocr::OcrItem> items{bottom, top_right, top_left};
    ppocr::sort_boxes_reading_order(items);

    check(items[0].text == "top_left");
    check(items[1].text == "top_right");
    check(items[2].text == "bottom");
}

void test_order_box_points_returns_clockwise_top_left_first() {
    const std::array<ppocr::Point, 4> unordered{
        ppocr::Point{110.0F, 60.0F},
        ppocr::Point{10.0F, 20.0F},
        ppocr::Point{90.0F, 110.0F},
        ppocr::Point{30.0F, 70.0F},
    };

    const auto ordered = ppocr::order_box_points(unordered);

    check(ordered[0].x == 10.0F && ordered[0].y == 20.0F);
    check(ordered[1].x == 110.0F && ordered[1].y == 60.0F);
    check(ordered[2].x == 90.0F && ordered[2].y == 110.0F);
    check(ordered[3].x == 30.0F && ordered[3].y == 70.0F);
}

void test_config_rejects_rec_direct_max_width_smaller_than_segment_width() {
    ppocr::Config config;
    config.api_key = "abc";
    config.rec_max_width = 1280;
    config.rec_direct_max_width = 960;

    bool threw = false;
    try {
        config.validate();
    } catch (const std::exception&) {
        threw = true;
    }

    check(threw);
}

void test_order_box_points_handles_diamond_without_duplicate_points() {
    const std::array<ppocr::Point, 4> diamond{
        ppocr::Point{50.0F, 0.0F},
        ppocr::Point{100.0F, 50.0F},
        ppocr::Point{50.0F, 100.0F},
        ppocr::Point{0.0F, 50.0F},
    };

    const auto ordered = ppocr::order_box_points(diamond);

    for (std::size_t i = 0; i < ordered.size(); ++i) {
        for (std::size_t j = i + 1; j < ordered.size(); ++j) {
            check(!(ordered[i].x == ordered[j].x && ordered[i].y == ordered[j].y), "ordered box contains duplicate points");
        }
    }
    check(ordered[0].x == 50.0F && ordered[0].y == 0.0F);
    check(ordered[1].x == 100.0F && ordered[1].y == 50.0F);
    check(ordered[2].x == 50.0F && ordered[2].y == 100.0F);
    check(ordered[3].x == 0.0F && ordered[3].y == 50.0F);
}

void test_expand_text_box_adds_padding_and_clamps_to_image_bounds() {
    const std::array<ppocr::Point, 4> box{
        ppocr::Point{1.0F, 1.0F},
        ppocr::Point{101.0F, 1.0F},
        ppocr::Point{101.0F, 31.0F},
        ppocr::Point{1.0F, 31.0F},
    };

    const auto expanded = ppocr::expand_text_box(box, 120.0F, 40.0F);

    check(expanded[0].x == 0.0F);
    check(expanded[0].y == 0.0F);
    check(expanded[1].x > box[1].x);
    check(expanded[2].y > box[2].y);
    for (const auto& point : expanded) {
        check(point.x >= 0.0F && point.x <= 119.0F);
        check(point.y >= 0.0F && point.y <= 39.0F);
    }
}

void test_merge_text_segments_removes_utf8_overlap() {
    const std::string chongqing = "\xE9\x87\x8D" "\xE5\xBA\x86";
    const std::string waimai = "\xE5\xA4\x96" "\xE5\x8D\x96";
    const std::string dingdan = "\xE8\xAE\xA2" "\xE5\x8D\x95";
    const std::string jine = "\xE9\x87\x91" "\xE9\xA2\x9D";
    const std::string yuan = "\xE5\x85\x83";
    const std::string jiezh = "\xE6\x88\xAA" "\xE6\xAD\xA2";
    const std::vector<std::string> segments{
        chongqing + waimai + dingdan,
        dingdan + jine + "8" + yuan,
        "8" + yuan + jiezh + "02:19",
    };

    const auto merged = ppocr::merge_text_segments(segments);

    check(merged == chongqing + waimai + dingdan + jine + "8" + yuan + jiezh + "02:19");
}

void test_orientation_retry_requires_confidence_margin_and_length_guard() {
    check(!ppocr::should_use_orientation_retry_result("abcdef", 0.70F, "abc", 0.80F));
    check(!ppocr::should_use_orientation_retry_result("abcdef", 0.70F, "abcdef", 0.74F));
    check(ppocr::should_use_orientation_retry_result("abcdef", 0.70F, "abcde", 0.76F));
    check(ppocr::should_use_orientation_retry_result("", 0.0F, "abc", 0.10F));
}

void test_startup_usage_text_contains_api_contract() {
    ppocr::Config config;
    config.listen_host = "0.0.0.0";
    config.port = 8080;
    config.api_key = "secret";

    const auto usage = ppocr::startup_usage_text(config);

    check(usage.find("GET /health") != std::string::npos);
    check(usage.find("POST /ocr") != std::string::npos);
    check(usage.find("X-API-Key: secret") != std::string::npos);
    check(usage.find("\"image_base64\"") != std::string::npos);
    check(usage.find("\"full_text\"") != std::string::npos);
    check(usage.find("\"items\"") != std::string::npos);
    check(usage.find("\"confidence\"") != std::string::npos);
    check(usage.find("\"box\"") != std::string::npos);
}

void test_startup_usage_text_describes_model_levels() {
    ppocr::Config config;
    config.listen_host = "0.0.0.0";
    config.port = 8080;
    config.api_key = "secret";

    const auto usage = ppocr::startup_usage_text(config);

    check(usage.find("Model levels:") != std::string::npos);
    check(usage.find("medium") != std::string::npos);
    check(usage.find("small") != std::string::npos);
    check(usage.find("tiny") != std::string::npos);
    check(usage.find("highest accuracy") != std::string::npos);
    check(usage.find("fastest") != std::string::npos);
}

void test_ocr_duration_log_format_contains_elapsed_ms() {
    const auto message = ppocr::format_ocr_duration_log(123);
    check(message.find("OCR inference completed in 123 ms") != std::string::npos);
}

} // namespace

int main() {
    run_test("test_default_config_is_created", test_default_config_is_created);
    run_test("test_config_reads_and_validates_rec_options", test_config_reads_and_validates_rec_options);
    run_test("test_config_rejects_invalid_rec_max_width", test_config_rejects_invalid_rec_max_width);
    run_test("test_config_rejects_rec_direct_max_width_smaller_than_segment_width", test_config_rejects_rec_direct_max_width_smaller_than_segment_width);
    run_test("test_config_rejects_invalid_model_level", test_config_rejects_invalid_model_level);
    run_test("test_base64_decodes_plain_and_data_url", test_base64_decodes_plain_and_data_url);
    run_test("test_base64_rejects_invalid_input", test_base64_rejects_invalid_input);
    run_test("test_request_queue_rejects_when_full", test_request_queue_rejects_when_full);
    run_test("test_ctc_decode_skips_blank_and_repeated_indices", test_ctc_decode_skips_blank_and_repeated_indices);
    run_test("test_boxes_sort_in_reading_order", test_boxes_sort_in_reading_order);
    run_test("test_boxes_sort_uses_dynamic_line_height_for_large_images", test_boxes_sort_uses_dynamic_line_height_for_large_images);
    run_test("test_order_box_points_returns_clockwise_top_left_first", test_order_box_points_returns_clockwise_top_left_first);
    run_test("test_order_box_points_handles_diamond_without_duplicate_points", test_order_box_points_handles_diamond_without_duplicate_points);
    run_test("test_expand_text_box_adds_padding_and_clamps_to_image_bounds", test_expand_text_box_adds_padding_and_clamps_to_image_bounds);
    run_test("test_merge_text_segments_removes_utf8_overlap", test_merge_text_segments_removes_utf8_overlap);
    run_test("test_orientation_retry_requires_confidence_margin_and_length_guard", test_orientation_retry_requires_confidence_margin_and_length_guard);
    run_test("test_startup_usage_text_contains_api_contract", test_startup_usage_text_contains_api_contract);
    run_test("test_startup_usage_text_describes_model_levels", test_startup_usage_text_describes_model_levels);
    run_test("test_ocr_duration_log_format_contains_elapsed_ms", test_ocr_duration_log_format_contains_elapsed_ms);
    return 0;
}
