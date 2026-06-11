#include "ppocr/http_server.h"

#include "ppocr/base64.h"
#include "ppocr/logger.h"
#include "ppocr/request_queue.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstring>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace ppocr {
namespace {

#if defined(_WIN32)
using socket_t = SOCKET;
constexpr socket_t invalid_socket_value = INVALID_SOCKET;
void close_socket(socket_t socket) {
    closesocket(socket);
}
#else
using socket_t = int;
constexpr socket_t invalid_socket_value = -1;
void close_socket(socket_t socket) {
    close(socket);
}
#endif

struct WsaSession {
    WsaSession() {
#if defined(_WIN32)
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
    }

    ~WsaSession() {
#if defined(_WIN32)
        WSACleanup();
#endif
    }
};

struct OcrJob {
    std::vector<std::uint8_t> bytes;
    std::promise<OcrResult> promise;
};

std::string reason_phrase(int status) {
    switch (status) {
    case 200:
        return "OK";
    case 400:
        return "Bad Request";
    case 401:
        return "Unauthorized";
    case 404:
        return "Not Found";
    case 405:
        return "Method Not Allowed";
    case 429:
        return "Too Many Requests";
    case 503:
        return "Service Unavailable";
    default:
        return "Internal Server Error";
    }
}

std::string json_response(int status, const nlohmann::json& body) {
    const auto payload = body.dump();
    std::ostringstream out;
    out << "HTTP/1.1 " << status << " " << reason_phrase(status) << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Content-Length: " << payload.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << payload;
    return out.str();
}

std::string read_request(socket_t client) {
    std::string request;
    char buffer[4096];
    while (request.find("\r\n\r\n") == std::string::npos) {
        const int received = recv(client, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            throw std::runtime_error("client disconnected while reading headers");
        }
        request.append(buffer, buffer + received);
        if (request.size() > 1024 * 1024) {
            throw std::runtime_error("request headers too large");
        }
    }

    const auto header_end = request.find("\r\n\r\n");
    const auto headers = request.substr(0, header_end);
    std::size_t content_length = 0;
    std::istringstream stream(headers);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string key = "Content-Length:";
        if (line.size() >= key.size() && _strnicmp(line.c_str(), key.c_str(), key.size()) == 0) {
            content_length = static_cast<std::size_t>(std::stoul(line.substr(key.size())));
        }
    }

    const auto body_start = header_end + 4;
    while (request.size() < body_start + content_length) {
        const int received = recv(client, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            throw std::runtime_error("client disconnected while reading body");
        }
        request.append(buffer, buffer + received);
    }
    return request;
}

std::string header_value(const std::string& headers, const std::string& name) {
    std::istringstream stream(headers);
    std::string line;
    const auto wanted = name + ":";
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.size() >= wanted.size() && _strnicmp(line.c_str(), wanted.c_str(), wanted.size()) == 0) {
            auto value = line.substr(wanted.size());
            while (!value.empty() && value.front() == ' ') {
                value.erase(value.begin());
            }
            return value;
        }
    }
    return {};
}

nlohmann::json ocr_result_to_json(const OcrResult& result) {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& item : result.items) {
        nlohmann::json box = nlohmann::json::array();
        for (const auto& point : item.box) {
            box.push_back({{"x", point.x}, {"y", point.y}});
        }
        items.push_back({
            {"text", item.text},
            {"confidence", item.confidence},
            {"box", box},
        });
    }
    return {{"full_text", result.full_text}, {"items", items}};
}

} // namespace

HttpServer::HttpServer(Config config, std::shared_ptr<OcrEngine> engine)
    : config_(std::move(config)), engine_(std::move(engine)) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::stop() {
    stopping_ = true;
}

void HttpServer::run() {
    WsaSession wsa;
    RequestQueue<std::shared_ptr<OcrJob>> queue(config_.queue_size);
    std::vector<std::thread> workers;
    for (std::size_t i = 0; i < config_.max_concurrent_requests; ++i) {
        workers.emplace_back([this, &queue] {
            std::shared_ptr<OcrJob> job;
            while (queue.wait_pop(job)) {
                try {
                    job->promise.set_value(engine_->recognize(job->bytes));
                } catch (...) {
                    job->promise.set_exception(std::current_exception());
                }
            }
        });
    }

    socket_t server = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == invalid_socket_value) {
        throw std::runtime_error("failed to create server socket");
    }

    int yes = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<unsigned short>(config_.port));
    if (config_.listen_host == "0.0.0.0") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else {
#if defined(_WIN32)
        if (InetPtonA(AF_INET, config_.listen_host.c_str(), &addr.sin_addr) != 1) {
            close_socket(server);
            throw std::runtime_error("listen_host must be an IPv4 address");
        }
#else
        if (inet_pton(AF_INET, config_.listen_host.c_str(), &addr.sin_addr) != 1) {
            close_socket(server);
            throw std::runtime_error("listen_host must be an IPv4 address");
        }
#endif
    }

    if (bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        close_socket(server);
        throw std::runtime_error("failed to bind HTTP socket");
    }
    if (listen(server, SOMAXCONN) != 0) {
        close_socket(server);
        throw std::runtime_error("failed to listen HTTP socket");
    }

    log::info("HTTP server listening on " + config_.listen_host + ":" + std::to_string(config_.port));

    while (!stopping_) {
        socket_t client = accept(server, nullptr, nullptr);
        if (client == invalid_socket_value) {
            if (!stopping_) {
                log::warn("accept failed");
            }
            continue;
        }

        std::thread([this, client, &queue] {
            std::string response;
            try {
                const auto request = read_request(client);
                const auto header_end = request.find("\r\n\r\n");
                const auto header_block = request.substr(0, header_end);
                const auto body = request.substr(header_end + 4);

                std::istringstream first_line_stream(header_block);
                std::string method;
                std::string path;
                first_line_stream >> method >> path;

                if (method == "GET" && path == "/health") {
                    const auto& runtime = engine_->runtime_info();
                    response = json_response(200, {
                        {"status", "ok"},
                        {"model_level", config_.model_level},
                        {"inference_mode", runtime.inference_mode},
                        {"gpu_enabled", runtime.gpu_enabled},
                        {"gpu_device_name", runtime.gpu_device_name},
                        {"fallback_reason", runtime.fallback_reason},
                    });
                } else if (method == "POST" && path == "/ocr") {
                    if (header_value(header_block, "X-API-Key") != config_.api_key) {
                        response = json_response(401, {{"error", "invalid API key"}});
                    } else {
                        const auto json = nlohmann::json::parse(body);
                        const auto image_base64 = json.at("image_base64").get<std::string>();
                        auto job = std::make_shared<OcrJob>();
                        job->bytes = decode_base64(image_base64);
                        auto future = job->promise.get_future();
                        if (!queue.try_push(job)) {
                            response = json_response(429, {{"error", "OCR queue is full"}});
                        } else {
                            response = json_response(200, ocr_result_to_json(future.get()));
                        }
                    }
                } else {
                    response = json_response(404, {{"error", "not found"}});
                }
            } catch (const nlohmann::json::exception& ex) {
                response = json_response(400, {{"error", ex.what()}});
            } catch (const std::invalid_argument& ex) {
                response = json_response(400, {{"error", ex.what()}});
            } catch (const std::runtime_error& ex) {
                response = json_response(503, {{"error", ex.what()}});
            } catch (const std::exception& ex) {
                response = json_response(500, {{"error", ex.what()}});
            }
            send(client, response.data(), static_cast<int>(response.size()), 0);
            close_socket(client);
        }).detach();
    }

    close_socket(server);
    queue.close();
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace ppocr
