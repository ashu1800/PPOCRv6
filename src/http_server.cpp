#include "ppocr/http_server.h"

#include "ppocr/base64.h"
#include "ppocr/http_request.h"
#include "ppocr/logger.h"
#include "ppocr/metrics.h"
#include "ppocr/request_queue.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
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

class PayloadTooLargeError final : public std::runtime_error {
public:
    explicit PayloadTooLargeError(const std::string& message) : std::runtime_error(message) {}
};

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
    std::string image_base64;
    std::promise<OcrResult> promise;
};

void set_socket_timeouts(socket_t socket) {
#if defined(_WIN32)
    constexpr DWORD timeout_ms = 30000;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
    timeval timeout{};
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
}

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
    case 413:
        return "Payload Too Large";
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

void send_all(socket_t client, const std::string& response) {
    std::size_t sent_total = 0;
    while (sent_total < response.size()) {
        const auto remaining = response.size() - sent_total;
        const int sent = send(client, response.data() + sent_total, static_cast<int>(std::min<std::size_t>(remaining, 64 * 1024)), 0);
        if (sent <= 0) {
            throw std::runtime_error("failed to send HTTP response");
        }
        sent_total += static_cast<std::size_t>(sent);
    }
}

struct RawHttpRequestHead {
    HttpRequestHead head;
    std::string initial_body;
};

RawHttpRequestHead read_request_head(socket_t client, std::size_t max_body_bytes) {
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
    auto head = parse_http_request_head(std::string_view(request).substr(0, header_end), max_body_bytes);
    return {std::move(head), request.substr(header_end + 4)};
}

std::string read_request_body(socket_t client, std::string body, std::size_t content_length, std::size_t max_body_bytes) {
    char buffer[4096];
    while (body.size() < content_length) {
        const int received = recv(client, buffer, sizeof(buffer), 0);
        if (received <= 0) {
            throw std::runtime_error("client disconnected while reading body");
        }
        body.append(buffer, buffer + received);
        if (body.size() > max_body_bytes) {
            throw PayloadTooLargeError("request body exceeds max_request_body_bytes");
        }
    }
    if (body.size() > content_length) {
        body.resize(content_length);
    }
    return body;
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

void handle_client(socket_t client, const Config& config, const std::shared_ptr<OcrEngine>& engine, RequestQueue<std::shared_ptr<OcrJob>>& queue) {
    set_socket_timeouts(client);
    std::string response;
    try {
        auto request = read_request_head(client, config.max_request_body_bytes);

        if (request.head.method == "GET" && request.head.path == "/health") {
            const auto& runtime = engine->runtime_info();
            response = json_response(200, {
                {"status", "ok"},
                {"model_level", config.model_level},
                {"inference_mode", runtime.inference_mode},
                {"gpu_enabled", runtime.gpu_enabled},
                {"gpu_device_name", runtime.gpu_device_name},
                {"fallback_reason", runtime.fallback_reason},
            });
        } else if (request.head.method == "POST" && request.head.path == "/ocr") {
            if (!should_read_http_body(request.head, config)) {
                response = json_response(401, {{"error", "invalid API key"}});
            } else {
                auto body = read_request_body(
                    client,
                    std::move(request.initial_body),
                    request.head.content_length,
                    config.max_request_body_bytes);
                std::future<OcrResult> future;
                bool queued = false;
                {
                    const auto json = nlohmann::json::parse(body);
                    auto image_base64 = json.at("image_base64").get<std::string>();
                    auto job = std::make_shared<OcrJob>();
                    job->image_base64 = std::move(image_base64);
                    future = job->promise.get_future();
                    queued = queue.try_push(job);
                }
                body.clear();
                body.shrink_to_fit();
                if (queued) {
                    response = json_response(200, ocr_result_to_json(future.get()));
                } else {
                    response = json_response(429, {{"error", "OCR queue is full"}});
                }
            }
        } else {
            response = json_response(404, {{"error", "not found"}});
        }
    } catch (const PayloadTooLargeError& ex) {
        response = json_response(413, {{"error", ex.what()}});
    } catch (const std::length_error& ex) {
        response = json_response(413, {{"error", ex.what()}});
    } catch (const nlohmann::json::exception& ex) {
        response = json_response(400, {{"error", ex.what()}});
    } catch (const std::invalid_argument& ex) {
        response = json_response(400, {{"error", ex.what()}});
    } catch (const std::runtime_error& ex) {
        response = json_response(503, {{"error", ex.what()}});
    } catch (const std::exception& ex) {
        response = json_response(500, {{"error", ex.what()}});
    }

    try {
        send_all(client, response);
    } catch (const std::exception& ex) {
        log::warn(ex.what());
    }
    close_socket(client);
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
    RequestQueue<socket_t> client_queue(std::max<std::size_t>(config_.queue_size, config_.max_concurrent_requests * 2));
    std::vector<std::thread> workers;
    std::vector<std::thread> client_workers;

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

    for (std::size_t i = 0; i < config_.max_concurrent_requests; ++i) {
        workers.emplace_back([this, &queue] {
            std::shared_ptr<OcrJob> job;
            while (queue.wait_pop(job)) {
                const auto started = std::chrono::steady_clock::now();
                try {
                    auto image_bytes = decode_base64(job->image_base64);
                    job->image_base64.clear();
                    job->image_base64.shrink_to_fit();
                    auto result = engine_->recognize(image_bytes);
                    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::steady_clock::now() - started)
                                             .count();
                    log::info(format_ocr_duration_log(elapsed));
                    job->promise.set_value(std::move(result));
                } catch (...) {
                    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::steady_clock::now() - started)
                                             .count();
                    log::warn("OCR inference failed after " + std::to_string(elapsed) + " ms");
                    job->promise.set_exception(std::current_exception());
                }
            }
        });
    }

    const std::size_t client_worker_count = std::max<std::size_t>(2, config_.max_concurrent_requests);
    for (std::size_t i = 0; i < client_worker_count; ++i) {
        client_workers.emplace_back([this, &queue, &client_queue] {
            socket_t client = invalid_socket_value;
            while (client_queue.wait_pop(client)) {
                handle_client(client, config_, engine_, queue);
            }
        });
    }

    while (!stopping_) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(server, &read_set);
        timeval timeout{};
        timeout.tv_sec = 0;
        timeout.tv_usec = 200000;
#if defined(_WIN32)
        const int ready = select(0, &read_set, nullptr, nullptr, &timeout);
#else
        const int ready = select(server + 1, &read_set, nullptr, nullptr, &timeout);
#endif
        if (ready == 0) {
            continue;
        }
        if (ready < 0) {
            if (!stopping_) {
                log::warn("select failed");
            }
            continue;
        }

        socket_t client = accept(server, nullptr, nullptr);
        if (client == invalid_socket_value) {
            if (!stopping_) {
                log::warn("accept failed");
            }
            continue;
        }

        set_socket_timeouts(client);
        if (!client_queue.try_push(client)) {
            try {
                send_all(client, json_response(429, {{"error", "HTTP connection queue is full"}}));
            } catch (const std::exception& ex) {
                log::warn(ex.what());
            }
            close_socket(client);
        }
    }

    close_socket(server);
    client_queue.close();
    for (auto& client_worker : client_workers) {
        if (client_worker.joinable()) {
            client_worker.join();
        }
    }
    queue.close();
    for (auto& worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace ppocr
