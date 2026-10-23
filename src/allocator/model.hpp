// One registered model and the runtimes built for it.
//
// A model may be loaded on more than one backend at once: the device
// selector can then send a request to whichever is free. Each backend's
// runtime runs one inference at a time -- a TFLite interpreter is not
// re-entrant -- so concurrency comes from different models and backends,
// not from one interpreter.

#pragma once

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "esca/inference/types.hpp"
#include "strategy.hpp"

namespace esca::inference {

enum class ModelState { Registered, Loading, Loaded, Failed };
const char *to_string(ModelState state);

/// The tensors a model takes and gives, as first loaded. Shapes may change
/// per context for models with dynamic inputs.
struct ModelMetadata {
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
};

struct RunOutcome {
    std::vector<TensorInfo> outputs; ///< shapes as produced, for dynamic outputs
};

class Model {
public:
    Model(ModelConfig config, StrategyFactory factory, int threads);

    const ModelConfig &config() const { return config_; }
    const std::string &name() const { return config_.name; }

    ModelState state() const;
    std::string last_error() const;

    /// Builds and loads the runtime for `backend`. Throws what the strategy
    /// throws; a BackendUnavailable marks the backend unusable for this
    /// model so the selector stops offering it.
    void load(Backend backend);
    bool loaded(Backend backend) const;
    std::vector<Backend> loaded_backends() const;
    bool unusable(Backend backend) const;

    /// Drops every runtime. The caller makes sure nothing is in flight.
    void unload();

    bool has_metadata() const;
    /// Throws ModelException if the model has never been loaded.
    ModelMetadata metadata() const;

    /// Runs on `backend`, which must be loaded. Serialised per backend.
    RunOutcome run(Backend backend, const std::vector<std::vector<int32_t>> &shapes,
                   const std::vector<ConstBuffer> &inputs,
                   const std::vector<MutableBuffer> &outputs);

    /// The tensors as they would be with these input shapes: what a context
    /// that set them needs its buffers sized for. `backend` must be loaded.
    ModelMetadata describe(Backend backend, const std::vector<std::vector<int32_t>> &shapes);

    /// Bytes held by the loaded runtimes; an estimate from the file size
    /// before the first load.
    size_t footprint() const;
    size_t estimate_load_bytes() const;

    std::chrono::steady_clock::time_point last_used() const;
    void touch();

    /// Requests queued or running on this model; a busy model is neither
    /// unloaded nor evicted.
    void acquire() { ++in_flight_; }
    void release() { --in_flight_; }
    int in_flight() const { return in_flight_.load(); }

private:
    struct Executor {
        std::unique_ptr<InferenceStrategy> strategy;
        std::mutex run_mutex; ///< one inference at a time per runtime
    };

    ModelConfig config_;
    StrategyFactory factory_;
    int threads_;

    /// Serialises loads, so two callers wanting the same backend build it
    /// once. Separate from mutex_, which the slow load must not hold.
    std::mutex load_mutex_;
    mutable std::mutex mutex_;
    std::map<Backend, std::shared_ptr<Executor>> executors_;
    std::set<Backend> unusable_;
    ModelState state_ = ModelState::Registered;
    std::string last_error_;
    ModelMetadata metadata_;
    bool has_metadata_ = false;
    std::chrono::steady_clock::time_point last_used_ = std::chrono::steady_clock::now();
    std::atomic<int> in_flight_ { 0 };
};

} // namespace esca::inference
