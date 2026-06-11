#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>

namespace ppocr {

template <typename T>
class RequestQueue {
public:
    explicit RequestQueue(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    bool try_push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= capacity_) {
            return false;
        }
        queue_.push(std::move(value));
        cv_.notify_one();
        return true;
    }

    bool wait_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    std::size_t capacity_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<T> queue_;
    bool closed_ = false;
};

} // namespace ppocr
