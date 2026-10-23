// libesca-resource-allocator: which models exist, which are loaded, and
// where each request runs.
//
// The model manager. It owns the models, decides when one is loaded or
// unloaded -- on demand, ahead of time, or under memory pressure -- and runs
// requests on the backend the device selector picks.

#pragma once

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "device_selector.hpp"
#include "memory_monitor.hpp"
#include "model.hpp"
#include "policies.hpp"
#include "strategy.hpp"

namespace esca::inference {

struct ModelSummary {
    std::string name;
    ModelState state = ModelState::Registered;
    std::vector<Backend> preferred;
    std::vector<Backend> loaded;
    std::string model_path;
    std::string last_error;
    size_t bytes = 0;
    int in_flight = 0;
};

struct ModelManagerOptions {
    /// Never let a load leave less than this available.
    size_t memory_reserve_bytes = 128u << 20;
    /// Unload idle models when PSI "some avg10" passes this, in percent.
    double pressure_threshold = 10.0;
    std::chrono::milliseconds watch_period { 2000 };
    unsigned cores = 4;
    /// Requests the CPU backend takes at once before it counts as busy.
    int cpu_slots = 2;
};

/// Called whenever a model changes state: name, state, backend involved.
using ModelStateListener =
    std::function<void(const std::string &, ModelState, const std::string &)>;

class ModelManager {
public:
    ModelManager(StrategyFactory factory, BackendProbe available,
                 std::unique_ptr<EvictionPolicy> eviction, std::shared_ptr<MemoryMonitor> memory,
                 ModelManagerOptions options = {});
    ~ModelManager();

    ModelManager(const ModelManager &) = delete;
    ModelManager &operator=(const ModelManager &) = delete;

    void set_listener(ModelStateListener listener);

    /// Throws ModelAlreadyRegistered. Preloads when the config asks.
    void register_model(const ModelConfig &config);
    /// Throws ModelNotFound, or ModelBusy while requests are in flight.
    void unregister_model(const std::string &name);
    bool contains(const std::string &name) const;
    std::vector<ModelSummary> list() const;
    ModelConfig config(const std::string &name) const;

    /// Loads on the first backend that works, making room first if memory
    /// is short. Returns the backend. Throws ModelLoadFailed, OutOfMemory or
    /// BackendUnavailable when none works.
    Backend load(const std::string &name);
    /// load() on the preload thread; failures are logged and reported to
    /// the listener, not thrown.
    void preload_async(const std::string &name);
    /// Throws ModelBusy while requests are in flight.
    void unload(const std::string &name);

    /// Tensor descriptions, loading the model first if it never was.
    ModelMetadata metadata(const std::string &name);

    /// Tensor descriptions for these input shapes, for a context resizing
    /// its buffers. Loads the model if needed.
    ModelMetadata describe(const std::string &name, const std::vector<std::vector<int32_t>> &shapes);

    /// Runs one request: picks a backend, loads it if needed (and falls back
    /// to the next one if it cannot be loaded), runs. `used` gets the backend.
    RunOutcome run(const std::string &name, const std::vector<std::vector<int32_t>> &shapes,
                   const std::vector<ConstBuffer> &inputs,
                   const std::vector<MutableBuffer> &outputs, Backend &used);

    /// Marks requests queued on a model, so it is not unloaded under them.
    void acquire(const std::string &name);
    void release(const std::string &name);

    void set_eviction_policy(std::unique_ptr<EvictionPolicy> policy);
    std::string eviction_policy() const;

    size_t loaded_bytes() const;
    const DeviceSelector &selector() const { return selector_; }

    /// One pass of the pressure watcher; public for tests.
    void relieve_pressure();

private:
    std::shared_ptr<Model> find(const std::string &name) const;
    Backend load_locked_out(const std::shared_ptr<Model> &model);
    void ensure_memory(const Model &model);
    std::vector<LoadedModel> loaded_models(const std::string &except) const;
    void notify(const std::string &name, ModelState state, const std::string &backend);
    void watch_loop();
    void preload_loop();

    StrategyFactory factory_;
    std::shared_ptr<MemoryMonitor> memory_;
    ModelManagerOptions options_;
    DeviceSelector selector_;

    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<Model>> models_;
    std::unique_ptr<EvictionPolicy> eviction_;
    ModelStateListener listener_;

    /// Loads and evictions take turns, so two loads cannot both count the
    /// same free memory.
    std::mutex memory_mutex_;

    std::mutex thread_mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    std::deque<std::string> preload_queue_;
    std::thread watcher_;
    std::thread preloader_;
};

} // namespace esca::inference
