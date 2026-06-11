#include "ppocr/metrics.h"

#include <string>

namespace ppocr {

std::string format_ocr_duration_log(std::int64_t elapsed_ms) {
    return "OCR inference completed in " + std::to_string(elapsed_ms) + " ms";
}

} // namespace ppocr
