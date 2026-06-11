#pragma once

#include "ppocr/config.h"
#include "ppocr/ocr_engine.h"

#include <atomic>
#include <memory>
#include <string>

namespace ppocr {

class HttpServer {
public:
    HttpServer(Config config, std::shared_ptr<OcrEngine> engine);
    ~HttpServer();

    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    void run();
    void stop();

private:
    Config config_;
    std::shared_ptr<OcrEngine> engine_;
    std::atomic<bool> stopping_{false};
};

} // namespace ppocr
