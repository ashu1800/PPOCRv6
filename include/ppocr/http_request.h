#pragma once

#include "ppocr/config.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace ppocr {

struct HttpRequestHead {
    std::string method;
    std::string path;
    std::string headers;
    std::size_t content_length = 0;
};

HttpRequestHead parse_http_request_head(std::string_view header_block, std::size_t max_body_bytes);
std::string http_header_value(std::string_view headers, std::string_view name);
bool should_read_http_body(const HttpRequestHead& head, const Config& config);

} // namespace ppocr
