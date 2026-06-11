#include "ppocr/config.h"
#include "ppocr/http_server.h"
#include "ppocr/logger.h"
#include "ppocr/ocr_engine.h"
#include "ppocr/usage.h"

#include <cstdlib>
#include <exception>
#include <filesystem>

int main() {
    try {
        const auto runtime_dir = std::filesystem::current_path();
        const auto config_path = ppocr::runtime_config_path();
        auto config = ppocr::Config::load_or_create(config_path);

        ppocr::log::info("Config path: " + config_path.string());
        ppocr::log::info("Model level: " + config.model_level);

        auto engine = ppocr::create_ocr_engine(config, runtime_dir);
        const auto& runtime = engine->runtime_info();
        ppocr::log::info("Inference mode: " + runtime.inference_mode);
        if (runtime.gpu_enabled) {
            ppocr::log::info("GPU device: " + runtime.gpu_device_name);
        } else if (!runtime.fallback_reason.empty()) {
            ppocr::log::warn("CPU fallback reason: " + runtime.fallback_reason);
        }
        ppocr::log::info(ppocr::startup_usage_text(config));

        ppocr::HttpServer server(config, engine);
        server.run();
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        ppocr::log::error(ex.what());
        return EXIT_FAILURE;
    }
}
