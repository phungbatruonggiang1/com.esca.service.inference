// A model that doubles its float input, and a memory gauge the test sets.
// Stand-ins for TensorFlow Lite and /proc/meminfo, so the allocator's
// decisions can be checked on any machine.

#pragma once

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>

#include "esca/inference/errors.hpp"
#include "memory_monitor.hpp"
#include "strategy.hpp"

namespace fake {

using namespace esca::inference;

/// Shared knobs, set by each test before it runs.
struct World {
    std::set<Backend> available { Backend::Cpu, Backend::Xnnpack };
    std::set<Backend> refuse;    ///< load() throws BackendUnavailable on these
    size_t footprint = 10u << 20;
    std::chrono::milliseconds run_delay { 0 };
    std::atomic<int> loads { 0 };
    /// What the loaded fakes hold; the fake memory gauge subtracts it, so
    /// unloading a model really does free memory.
    std::atomic<size_t> loaded_bytes { 0 };
    std::atomic<int> runs { 0 };
    std::mutex order_mutex;
    std::vector<std::string> order; ///< models in the order they ran
};

inline World &world()
{
    static World w;
    return w;
}

inline void reset()
{
    World &w = world();
    w.available = { Backend::Cpu, Backend::Xnnpack };
    w.refuse.clear();
    w.footprint = 10u << 20;
    w.run_delay = std::chrono::milliseconds(0);
    w.loads = 0;
    w.loaded_bytes = 0;
    w.runs = 0;
    std::lock_guard<std::mutex> lock(w.order_mutex);
    w.order.clear();
}

class Strategy : public InferenceStrategy {
public:
    Strategy(std::string name, Backend backend) : name_(std::move(name)), backend_(backend) {}

    Backend backend() const override { return backend_; }
    void load() override
    {
        if (world().refuse.count(backend_))
            raise(ErrorCode::BackendUnavailable, "fake refuses " + std::string(to_string(backend_)));
        ++world().loads;
        world().loaded_bytes += world().footprint;
        loaded_ = true;
    }
    void unload() override
    {
        if (loaded_)
            world().loaded_bytes -= world().footprint;
        loaded_ = false;
    }
    bool loaded() const override { return loaded_; }
    std::vector<TensorInfo> inputs() const override { return { tensor("in") }; }
    std::vector<TensorInfo> outputs() const override { return { tensor("out") }; }
    void resize_inputs(const std::vector<std::vector<int32_t>> &) override {}
    void run(const std::vector<ConstBuffer> &in, const std::vector<MutableBuffer> &out) override
    {
        if (world().run_delay.count())
            std::this_thread::sleep_for(world().run_delay);
        {
            std::lock_guard<std::mutex> lock(world().order_mutex);
            world().order.push_back(name_);
        }
        ++world().runs;
        float v[4];
        std::memcpy(v, in.at(0).data, sizeof(v));
        for (float &x : v)
            x *= 2;
        std::memcpy(out.at(0).data, v, sizeof(v));
    }
    size_t memory_footprint() const override { return loaded_ ? world().footprint : 0; }

private:
    static TensorInfo tensor(const char *name)
    {
        return TensorInfo { name, DataType::Float32, { 1, 4 }, 0.0, 0, 16 };
    }
    std::string name_;
    Backend backend_;
    bool loaded_ = false;
};

inline StrategyFactory factory()
{
    return [](const ModelConfig &config, Backend backend, int) {
        return std::make_unique<Strategy>(config.name, backend);
    };
}

inline BackendProbe probe()
{
    return [](Backend b) { return world().available.count(b) != 0; };
}

class Memory : public MemoryMonitor {
public:
    size_t available_bytes() const override
    {
        const size_t used = world().loaded_bytes.load();
        return capacity > used ? capacity - used : 0;
    }
    std::optional<double> pressure() const override { return std::nullopt; }
    /// Free memory with nothing loaded.
    std::atomic<size_t> capacity { size_t { 4096 } << 20 };
};

} // namespace fake
