#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace ppocr {

struct Config {
    std::string model_level = "small";
    std::string listen_host = "0.0.0.0";
    int port = 8080;
    std::string api_key;
    bool prefer_gpu = true;
    int gpu_device = 0;
    std::size_t max_concurrent_requests = 2;
    std::size_t queue_size = 8;
    std::size_t ncnn_threads_per_request = 1;
    std::size_t max_request_body_bytes = 16 * 1024 * 1024;
    std::size_t rec_max_width = 960;
    std::size_t rec_direct_max_width = 4096;
    bool enable_orientation_retry = true;

    static Config load_or_create(const std::filesystem::path& path);
    void validate() const;
};

std::filesystem::path runtime_config_path();

} // namespace ppocr
