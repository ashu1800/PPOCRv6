#include "ppocr/base64.h"

#include <array>
#include <stdexcept>

namespace ppocr {
namespace {

std::array<int, 256> make_decode_table() {
    std::array<int, 256> table{};
    table.fill(-1);
    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (std::size_t i = 0; i < alphabet.size(); ++i) {
        table[static_cast<unsigned char>(alphabet[i])] = static_cast<int>(i);
    }
    return table;
}

std::string strip_data_url_prefix(const std::string& input) {
    const auto comma = input.find(',');
    if (input.rfind("data:", 0) == 0 && comma != std::string::npos) {
        return input.substr(comma + 1);
    }
    return input;
}

} // namespace

std::vector<std::uint8_t> decode_base64(const std::string& input) {
    static const auto table = make_decode_table();
    const auto encoded = strip_data_url_prefix(input);

    std::vector<std::uint8_t> out;
    int value = 0;
    int bits = -8;
    bool saw_padding = false;

    for (const unsigned char c : encoded) {
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            continue;
        }
        if (c == '=') {
            saw_padding = true;
            continue;
        }
        if (saw_padding) {
            throw std::invalid_argument("invalid base64: data after padding");
        }
        const int decoded = table[c];
        if (decoded < 0) {
            throw std::invalid_argument("invalid base64 character");
        }
        value = (value << 6) + decoded;
        bits += 6;
        if (bits >= 0) {
            out.push_back(static_cast<std::uint8_t>((value >> bits) & 0xFF));
            bits -= 8;
        }
    }

    if (out.empty() && !encoded.empty()) {
        throw std::invalid_argument("invalid base64 payload");
    }
    return out;
}

} // namespace ppocr
