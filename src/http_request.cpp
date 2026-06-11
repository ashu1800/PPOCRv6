#include "ppocr/http_request.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace ppocr {
namespace {

bool is_ows(char value) {
    return value == ' ' || value == '\t' || value == '\r';
}

bool ascii_iequals(std::string_view lhs, std::string_view rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        const auto left = static_cast<unsigned char>(lhs[i]);
        const auto right = static_cast<unsigned char>(rhs[i]);
        if (std::tolower(left) != std::tolower(right)) {
            return false;
        }
    }
    return true;
}

std::string trim_ows(std::string_view value) {
    while (!value.empty() && is_ows(value.front())) {
        value.remove_prefix(1);
    }
    while (!value.empty() && is_ows(value.back())) {
        value.remove_suffix(1);
    }
    return std::string(value);
}

} // namespace

HttpRequestHead parse_http_request_head(std::string_view header_block, std::size_t max_body_bytes) {
    HttpRequestHead head;
    head.headers = std::string(header_block);

    std::istringstream stream(head.headers);
    std::string line;
    if (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::istringstream first_line(line);
        first_line >> head.method >> head.path;
    }

    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        if (ascii_iequals(std::string_view(line).substr(0, colon), "Content-Length")) {
            const auto value = trim_ows(std::string_view(line).substr(colon + 1));
            std::size_t consumed = 0;
            const auto parsed = std::stoull(value, &consumed);
            if (consumed != value.size()) {
                throw std::invalid_argument("invalid Content-Length");
            }
            if (parsed > max_body_bytes) {
                throw std::length_error("request body exceeds max_request_body_bytes");
            }
            head.content_length = static_cast<std::size_t>(parsed);
        }
    }

    return head;
}

std::string http_header_value(std::string_view headers, std::string_view name) {
    std::istringstream stream{std::string(headers)};
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        if (ascii_iequals(std::string_view(line).substr(0, colon), name)) {
            return trim_ows(std::string_view(line).substr(colon + 1));
        }
    }
    return {};
}

bool should_read_http_body(const HttpRequestHead& head, const Config& config) {
    return head.method == "POST" && head.path == "/ocr" &&
           http_header_value(head.headers, "X-API-Key") == config.api_key;
}

} // namespace ppocr
