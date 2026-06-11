#include "ppocr/config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <random>
#include <set>
#include <stdexcept>

namespace ppocr {
namespace {

std::string generate_api_key() {
    static constexpr char alphabet[] = "0123456789abcdef";
    std::random_device rd;
    std::uniform_int_distribution<int> dist(0, 15);

    std::string key;
    key.reserve(32);
    for (int i = 0; i < 32; ++i) {
        key.push_back(alphabet[dist(rd)]);
    }
    return key;
}

nlohmann::json to_json(const Config& config) {
    return nlohmann::json{
        {"model_level", config.model_level},
        {"listen_host", config.listen_host},
        {"port", config.port},
        {"api_key", config.api_key},
        {"prefer_gpu", config.prefer_gpu},
        {"gpu_device", config.gpu_device},
        {"max_concurrent_requests", config.max_concurrent_requests},
        {"queue_size", config.queue_size},
        {"max_request_body_bytes", config.max_request_body_bytes},
        {"rec_max_width", config.rec_max_width},
        {"enable_orientation_retry", config.enable_orientation_retry},
    };
}

Config from_json(const nlohmann::json& json) {
    Config config;
    config.model_level = json.value("model_level", config.model_level);
    config.listen_host = json.value("listen_host", config.listen_host);
    config.port = json.value("port", config.port);
    config.api_key = json.value("api_key", std::string{});
    config.prefer_gpu = json.value("prefer_gpu", config.prefer_gpu);
    config.gpu_device = json.value("gpu_device", config.gpu_device);
    config.max_concurrent_requests = json.value("max_concurrent_requests", config.max_concurrent_requests);
    config.queue_size = json.value("queue_size", config.queue_size);
    config.max_request_body_bytes = json.value("max_request_body_bytes", config.max_request_body_bytes);
    config.rec_max_width = json.value("rec_max_width", config.rec_max_width);
    config.enable_orientation_retry = json.value("enable_orientation_retry", config.enable_orientation_retry);
    if (config.api_key.empty()) {
        config.api_key = generate_api_key();
    }
    return config;
}

} // namespace

std::filesystem::path runtime_config_path() {
    return std::filesystem::current_path() / "config.json";
}

Config Config::load_or_create(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        Config config;
        config.api_key = generate_api_key();
        config.validate();

        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to create config file: " + path.string());
        }
        out << to_json(config).dump(4);
        return config;
    }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open config file: " + path.string());
    }

    nlohmann::json json;
    in >> json;
    auto config = from_json(json);
    config.validate();
    return config;
}

void Config::validate() const {
    const std::set<std::string> allowed{"medium", "small", "tiny"};
    if (allowed.count(model_level) == 0) {
        throw std::invalid_argument("model_level must be one of medium, small, tiny");
    }
    if (listen_host.empty()) {
        throw std::invalid_argument("listen_host must not be empty");
    }
    if (port <= 0 || port > 65535) {
        throw std::invalid_argument("port must be in range 1..65535");
    }
    if (api_key.empty()) {
        throw std::invalid_argument("api_key must not be empty");
    }
    if (gpu_device < 0) {
        throw std::invalid_argument("gpu_device must be >= 0");
    }
    if (max_concurrent_requests == 0) {
        throw std::invalid_argument("max_concurrent_requests must be > 0");
    }
    if (queue_size < max_concurrent_requests) {
        throw std::invalid_argument("queue_size must be >= max_concurrent_requests");
    }
    if (max_request_body_bytes < 1024 * 1024) {
        throw std::invalid_argument("max_request_body_bytes must be at least 1048576");
    }
    if (rec_max_width < 320 || rec_max_width > 4096) {
        throw std::invalid_argument("rec_max_width must be in range 320..4096");
    }
}

} // namespace ppocr
