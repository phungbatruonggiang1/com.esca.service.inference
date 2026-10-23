#include "scheduler.hpp"

#include <algorithm>
#include <chrono>

#include "logger.hpp"

namespace esca::inference {

namespace {
constexpr size_t kRecent = 256;

double ms_since(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
} // namespace

InferenceScheduler::InferenceScheduler(ModelManager &models, std::unique_ptr<SchedulingPolicy> policy,
                                       unsigned workers)
    : models_(models),
      policy_(policy ? std::move(policy) : std::make_unique<FifoSchedulingPolicy>())
{
    recent_.reserve(kRecent);
    for (unsigned i = 0; i < std::max(1u, workers); ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

InferenceScheduler::~InferenceScheduler()
{
    std::map<uint64_t, InferenceRequest> abandoned;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        abandoned.swap(pending_);
        queue_.clear();
    }
    wake_.notify_all();
    for (std::thread &t : workers_)
        t.join();

    // Anything still queued is answered, so no caller waits forever.
    for (auto &[id, request] : abandoned) {
        models_.release(request.model);
        if (request.done)
            {
            InferenceResult stopped;
            stopped.code = ErrorCode::ServiceUnavailable;
            stopped.message = "scheduler stopped";
            request.done(stopped);
        }
    }
}

uint64_t InferenceScheduler::submit(InferenceRequest request)
{
    // Counted against the model before it is queued: from here until it
    // finishes, the model may not be unloaded under it.
    models_.acquire(request.model);

    uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        id = next_id_++;
        queue_.push_back(QueuedRequest { id, request.model, request.priority,
                                         std::chrono::steady_clock::now() });
        pending_.emplace(id, std::move(request));
    }
    wake_.notify_one();
    return id;
}

InferenceResult InferenceScheduler::run(InferenceRequest request)
{
    auto promise = std::make_shared<std::promise<InferenceResult>>();
    std::future<InferenceResult> result = promise->get_future();
    request.done = [promise](const InferenceResult &r) { promise->set_value(r); };
    submit(std::move(request));
    return result.get();
}

void InferenceScheduler::worker_loop()
{
    for (;;) {
        QueuedRequest meta;
        InferenceRequest request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_)
                return;
            const size_t pick = std::min(policy_->next(queue_), queue_.size() - 1);
            meta = queue_[pick];
            queue_.erase(queue_.begin() + static_cast<long>(pick));
            auto it = pending_.find(meta.id);
            request = std::move(it->second);
            pending_.erase(it);
            policy_->on_dispatch(meta);
            ++running_;
        }

        InferenceResult result;
        result.wait_ms = ms_since(meta.enqueued);
        const auto started = std::chrono::steady_clock::now();
        try {
            RunOutcome outcome =
                models_.run(request.model, request.shapes, request.inputs, request.outputs, result.backend);
            result.outputs = std::move(outcome.outputs);
        } catch (const InferenceException &e) {
            result.code = e.code();
            result.message = e.what();
        } catch (const std::exception &e) {
            result.code = ErrorCode::InferenceFailed;
            result.message = e.what();
        }
        result.run_ms = ms_since(started);
        models_.release(request.model);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --running_;
            policy_->on_complete(meta, result.run_ms);
        }
        record(result.code == ErrorCode::Ok, result.run_ms, result.wait_ms);
        if (result.code != ErrorCode::Ok)
            ESCA_LOG_WARN("inference on %s failed: %s", request.model.c_str(), result.message.c_str());

        if (request.done)
            request.done(result);
    }
}

void InferenceScheduler::record(bool ok, double run_ms, double wait_ms)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ok) {
        ++failed_;
        return;
    }
    ++completed_;
    total_ms_ += run_ms;
    total_wait_ms_ += wait_ms;
    if (recent_.size() < kRecent)
        recent_.push_back(run_ms);
    else
        recent_[recent_next_] = run_ms;
    recent_next_ = (recent_next_ + 1) % kRecent;
}

SchedulerStats InferenceScheduler::stats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    SchedulerStats s;
    s.queued = queue_.size();
    s.running = running_;
    s.completed = completed_;
    s.failed = failed_;
    s.policy = policy_->name();
    s.workers = static_cast<unsigned>(workers_.size());
    if (completed_ > 0) {
        s.avg_ms = total_ms_ / static_cast<double>(completed_);
        s.avg_wait_ms = total_wait_ms_ / static_cast<double>(completed_);
    }
    if (!recent_.empty()) {
        std::vector<double> sorted = recent_;
        std::sort(sorted.begin(), sorted.end());
        s.p95_ms = sorted[std::min(sorted.size() - 1, (sorted.size() * 95) / 100)];
    }
    return s;
}

void InferenceScheduler::set_policy(std::unique_ptr<SchedulingPolicy> policy)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (policy)
        policy_ = std::move(policy);
}

std::string InferenceScheduler::policy() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return policy_->name();
}

} // namespace esca::inference
