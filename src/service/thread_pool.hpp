// A fixed set of worker threads for work the bus thread must not wait on:
// loading a model to create a context, answering an asynchronous call.

#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace esca::inference {

class ThreadPool {
public:
    explicit ThreadPool(unsigned threads)
    {
        for (unsigned i = 0; i < (threads == 0 ? 1 : threads); ++i)
            threads_.emplace_back([this] { loop(); });
    }

    /// Finishes what is queued, then stops.
    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (std::thread &t : threads_)
            t.join();
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    void post(std::function<void()> job)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(std::move(job));
        }
        wake_.notify_one();
    }

private:
    void loop()
    {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
                if (jobs_.empty())
                    return; // stopping, and nothing left
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
    }

    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::function<void()>> jobs_;
    bool stopping_ = false;
    std::vector<std::thread> threads_;
};

} // namespace esca::inference
