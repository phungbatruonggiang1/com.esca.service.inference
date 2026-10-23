// The two decisions the allocator lets be swapped: which loaded model to
// give up when memory runs short, and which queued request runs next.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace esca::inference {

/// What an eviction policy sees of a loaded model.
struct LoadedModel {
    std::string name;
    size_t bytes = 0;
    std::chrono::steady_clock::time_point last_used;
    /// Inferences queued or running on it. A busy model is never a victim.
    int in_flight = 0;
};

class EvictionPolicy {
public:
    virtual ~EvictionPolicy() = default;
    virtual const char *name() const = 0;
    /// Models to unload so that at least `bytes_needed` come free. May return
    /// fewer than needed; the caller then reports OutOfMemory.
    virtual std::vector<std::string> select_victims(const std::vector<LoadedModel> &loaded,
                                                    size_t bytes_needed) const = 0;
};

/// The default: never unload anything on its own.
class NoEvictionPolicy : public EvictionPolicy {
public:
    const char *name() const override { return "none"; }
    std::vector<std::string> select_victims(const std::vector<LoadedModel> &,
                                            size_t) const override
    {
        return {};
    }
};

/// Least recently used first, idle models only. Unloading costs the next
/// caller a reload, so the one least likely to be called again goes first.
class LruEvictionPolicy : public EvictionPolicy {
public:
    const char *name() const override { return "lru"; }
    std::vector<std::string> select_victims(const std::vector<LoadedModel> &loaded,
                                            size_t bytes_needed) const override;
};

std::unique_ptr<EvictionPolicy> make_eviction_policy(const std::string &name);

/// What a scheduling policy sees of a queued request.
struct QueuedRequest {
    uint64_t id = 0;
    std::string model;
    int priority = 0; ///< larger runs sooner, for policies that look
    std::chrono::steady_clock::time_point enqueued;
};

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual const char *name() const = 0;
    /// Index into `queue` of the request to dispatch next; queue is not empty.
    virtual size_t next(const std::deque<QueuedRequest> &queue) = 0;
    virtual void on_dispatch(const QueuedRequest &) {}
    virtual void on_complete(const QueuedRequest &, double /*ms*/) {}
};

/// The default: arrival order.
class FifoSchedulingPolicy : public SchedulingPolicy {
public:
    const char *name() const override { return "fifo"; }
    size_t next(const std::deque<QueuedRequest> &) override { return 0; }
};

/// Highest priority first, arrival order among equals. A request that has
/// waited `aging` gains one priority level per period, so a steady stream of
/// urgent work cannot starve the rest forever.
class PrioritySchedulingPolicy : public SchedulingPolicy {
public:
    explicit PrioritySchedulingPolicy(std::chrono::milliseconds aging = std::chrono::milliseconds(500))
        : aging_(aging)
    {
    }
    const char *name() const override { return "priority"; }
    size_t next(const std::deque<QueuedRequest> &queue) override;

private:
    std::chrono::milliseconds aging_;
};

std::unique_ptr<SchedulingPolicy> make_scheduling_policy(const std::string &name);

} // namespace esca::inference
