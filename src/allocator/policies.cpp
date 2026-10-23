#include "policies.hpp"

#include <algorithm>

namespace esca::inference {

std::vector<std::string> LruEvictionPolicy::select_victims(const std::vector<LoadedModel> &loaded,
                                                           size_t bytes_needed) const
{
    std::vector<const LoadedModel *> idle;
    for (const LoadedModel &m : loaded)
        if (m.in_flight == 0)
            idle.push_back(&m);
    std::sort(idle.begin(), idle.end(),
              [](const LoadedModel *a, const LoadedModel *b) { return a->last_used < b->last_used; });

    std::vector<std::string> victims;
    size_t freed = 0;
    for (const LoadedModel *m : idle) {
        if (freed >= bytes_needed)
            break;
        victims.push_back(m->name);
        freed += m->bytes;
    }
    return victims;
}

std::unique_ptr<EvictionPolicy> make_eviction_policy(const std::string &name)
{
    if (name == "lru")
        return std::make_unique<LruEvictionPolicy>();
    if (name == "none")
        return std::make_unique<NoEvictionPolicy>();
    return nullptr;
}

size_t PrioritySchedulingPolicy::next(const std::deque<QueuedRequest> &queue)
{
    const auto now = std::chrono::steady_clock::now();
    auto effective = [&](const QueuedRequest &r) {
        const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(now - r.enqueued);
        return r.priority + (aging_.count() > 0 ? static_cast<int>(waited / aging_) : 0);
    };
    size_t best = 0;
    for (size_t i = 1; i < queue.size(); ++i)
        // Strictly greater: among equals the earlier arrival keeps its place.
        if (effective(queue[i]) > effective(queue[best]))
            best = i;
    return best;
}

std::unique_ptr<SchedulingPolicy> make_scheduling_policy(const std::string &name)
{
    if (name == "fifo")
        return std::make_unique<FifoSchedulingPolicy>();
    if (name == "priority")
        return std::make_unique<PrioritySchedulingPolicy>();
    return nullptr;
}

} // namespace esca::inference
