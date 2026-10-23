#include "device_selector.hpp"

#include "esca/inference/errors.hpp"

namespace esca::inference {

DeviceSelector::DeviceSelector(BackendProbe available, int cpu_slots)
    : available_(std::move(available)), cpu_slots_(cpu_slots < 1 ? 1 : cpu_slots)
{
}

std::vector<Backend> DeviceSelector::candidates(const Model &model) const
{
    std::vector<Backend> out;
    for (Backend b : model.config().backends) {
        bool seen = false;
        for (Backend o : out)
            seen = seen || o == b;
        if (!seen && available_(b) && !model.unusable(b))
            out.push_back(b);
    }
    return out;
}

Backend DeviceSelector::select(const Model &model) const
{
    const std::vector<Backend> options = candidates(model);
    if (options.empty())
        raise(ErrorCode::BackendUnavailable,
              model.name() + ": none of its backends is available on this machine");
    for (Backend b : options)
        if (free(b))
            return b;
    return options.front();
}

void DeviceSelector::acquire(Backend backend) { ++busy_[index(backend)]; }

void DeviceSelector::release(Backend backend) { --busy_[index(backend)]; }

int DeviceSelector::busy(Backend backend) const { return busy_[index(backend)].load(); }

bool DeviceSelector::free(Backend backend) const
{
    // XNNPACK spreads one inference over every thread it was given, so it is
    // "busy" with one, like an accelerator.
    const int slots = backend == Backend::Cpu ? cpu_slots_ : 1;
    return busy(backend) < slots;
}

} // namespace esca::inference
