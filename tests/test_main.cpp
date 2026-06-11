#include "ppocr/base64.h"
#include "ppocr/config.h"
#include "ppocr/ocr_postprocess.h"
#include "ppocr/request_queue.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

void test_default_config_is_created() {
    const auto dir = std::filesystem::temp_directory_path() / "ppocrv6_config_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const auto path = dir / "config.json";
    const auto config = ppocr::Config::load_or_create(path);

    assert(std::filesystem::exists(path));
    assert(config.model_level == "small");
    assert(config.listen_host == "0.0.0.0");
    assert(config.port == 8080);
    assert(config.prefer_gpu);
    assert(!config.api_key.empty());
    assert(config.max_concurrent_requests > 0);
    assert(config.queue_size >= config.max_concurrent_requests);

    std::filesystem::remove_all(dir);
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

    assert(threw);
    std::filesystem::remove_all(dir);
}

void test_base64_decodes_plain_and_data_url() {
    const auto plain = ppocr::decode_base64("SGVsbG8=");
    assert(std::string(plain.begin(), plain.end()) == "Hello");

    const auto data_url = ppocr::decode_base64("data:image/png;base64,SGVsbG8=");
    assert(std::string(data_url.begin(), data_url.end()) == "Hello");
}

void test_base64_rejects_invalid_input() {
    bool threw = false;
    try {
        (void)ppocr::decode_base64("%%%%");
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
}

void test_request_queue_rejects_when_full() {
    ppocr::RequestQueue<int> queue(1);
    assert(queue.try_push(1));
    assert(!queue.try_push(2));

    int value = 0;
    assert(queue.wait_pop(value));
    assert(value == 1);
}

void test_ctc_decode_skips_blank_and_repeated_indices() {
    const std::vector<std::string> keys{"A", "B", "C"};
    const auto decoded = ppocr::ctc_decode({1, 1, 0, 2, 2, 3}, {0.9F, 0.8F, 0.4F, 0.7F, 0.6F, 0.5F}, keys);

    assert(decoded.text == "ABC");
    assert(decoded.confidence > 0.0F);
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

    assert(items[0].text == "top_left");
    assert(items[1].text == "top_right");
    assert(items[2].text == "bottom");
}

} // namespace

int main() {
    test_default_config_is_created();
    test_config_rejects_invalid_model_level();
    test_base64_decodes_plain_and_data_url();
    test_base64_rejects_invalid_input();
    test_request_queue_rejects_when_full();
    test_ctc_decode_skips_blank_and_repeated_indices();
    test_boxes_sort_in_reading_order();
    return 0;
}
