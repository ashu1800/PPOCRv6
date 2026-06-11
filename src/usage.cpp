#include "ppocr/usage.h"

#include <sstream>

namespace ppocr {

std::string startup_usage_text(const Config& config) {
    std::ostringstream out;
    out << "API usage:\n"
        << "  Health: GET /health\n"
        << "  OCR: POST /ocr\n"
        << "  URL: http://" << config.listen_host << ":" << config.port << "/ocr\n"
        << "  Required header: X-API-Key: " << config.api_key << "\n"
        << "  Request JSON: {\"image_base64\":\"<base64 image or data URL>\"}\n"
        << "  Success response JSON: {\"full_text\":\"...\",\"items\":[{\"text\":\"...\",\"confidence\":0.98,\"box\":[{\"x\":0,\"y\":0},{\"x\":1,\"y\":0},{\"x\":1,\"y\":1},{\"x\":0,\"y\":1}]}]}\n"
        << "  Error response JSON: {\"error\":\"message\"}\n"
        << "Model levels:\n"
        << "  medium: highest accuracy, slowest speed; best for small text, dense screenshots, and quality-first OCR.\n"
        << "  small: balanced accuracy and speed; recommended default for common OCR service workloads.\n"
        << "  tiny: fastest speed, lowest accuracy; best for latency-first requests with clear, large text.";
    return out.str();
}

} // namespace ppocr
