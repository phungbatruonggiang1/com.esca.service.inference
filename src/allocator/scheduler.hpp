// Queues inference requests and runs them on worker threads.
//
// The inference scheduler: submit-and-wait or submit-and-callback, the
// order decided by a swappable SchedulingPolicy, with queue and latency
// statistics for the service to report.

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "esca/inference/errors.hpp"
#include "model_manager.hpp"
#include "policies.hpp"

namespace esca::inference {

struct InferenceResult {
    ErrorCode code = ErrorCode::Ok;
    std::string message;
    Backend backend = Backend::Cpu;
    double run_ms = 0;  ///< the inference itself
    double wait_ms = 0; ///< time spent queued
    std::vector<TensorInfo> outputs;
};

struct InferenceRequest {
    std::string model;
    int priority = 0;
    std::vector<std::vector<int32_t>> shapes;
    std::vector<ConstBuffer> inputs;
    std::vector<MutableBuffer> outputs;
    /// Called on a worker thread when the request is done, success or not.
    std::function<void(const InferenceResult &)> done;
};

struct SchedulerStats {
    size_t queued = 0;
    size_t running = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    double avg_ms = 0;
    double p95_ms = 0;
    double avg_wait_ms = 0;
    std::string policy;
    unsigned workers = 0;
};

class InferenceScheduler {
public:
    InferenceScheduler(ModelManager &models, std::unique_ptr<SchedulingPolicy> policy, unsigned workers);
    ~InferenceScheduler();

    InferenceScheduler(const InferenceScheduler &) = delete;
    InferenceScheduler &operator=(const InferenceScheduler &) = delete;

    /// Queues the request and returns its id. Throws ModelNotFound at once
    /// for an unknown model rather than failing later on a worker.
    uint64_t submit(InferenceRequest request);

    /// Queues and waits. `request.done` is ignored.
    InferenceResult run(InferenceRequest request);

    SchedulerStats stats() const;
    void set_policy(std::unique_ptr<SchedulingPolicy> policy);
    std::string policy() const;

private:
    struct Entry {
        QueuedRequest meta;
        InferenceRequest request;
    };

    void worker_loop();
    void record(bool ok, double run_ms, double wait_ms);

    ModelManager &models_;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::unique_ptr<SchedulingPolicy> policy_;
    std::deque<QueuedRequest> queue_;
    std::map<uint64_t, InferenceRequest> pending_;
    uint64_t next_id_ = 1;
    bool stopping_ = false;
    size_t running_ = 0;

    uint64_t completed_ = 0;
    uint64_t failed_ = 0;
    double total_ms_ = 0;
    double total_wait_ms_ = 0;
    std::vector<double> recent_; ///< last latencies, a ring, for the p95
    size_t recent_next_ = 0;

    std::vector<std::thread> workers_;
};

} // namespace esca::inference
