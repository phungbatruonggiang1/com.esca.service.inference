// Which backend a request runs on.
//
// Chosen per request, not per model: a model allowed on the NPU and the CPU
// goes to the NPU while it is free and to the CPU while another model holds
// the NPU, instead of queueing behind it. Busy is counted across all models,
// because the accelerator is shared, not the model.

#pragma once

#include <array>
#include <atomic>
#include <vector>

#include "esca/inference/types.hpp"
#include "model.hpp"
#include "strategy.hpp"

namespace esca::inference {

class DeviceSelector {
public:
    /// `cpu_slots` is how many requests the CPU backend takes at once before
    /// it counts as busy; accelerators take one.
    DeviceSelector(BackendProbe available, int cpu_slots);

    /// The model's preferences, keeping only backends this machine has and
    /// the model has not failed on.
    std::vector<Backend> candidates(const Model &model) const;

    /// The first candidate with a free slot; if every one is busy, the first
    /// candidate, to queue behind. Throws BackendUnavailable when no
    /// candidate is left at all.
    Backend select(const Model &model) const;

    void acquire(Backend backend);
    void release(Backend backend);
    int busy(Backend backend) const;
    bool free(Backend backend) const;

private:
    static size_t index(Backend backend) { return static_cast<size_t>(backend); }

    BackendProbe available_;
    int cpu_slots_;
    std::array<std::atomic<int>, 4> busy_ {};
};

} // namespace esca::inference
