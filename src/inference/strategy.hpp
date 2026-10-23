// libesca-inference: one model on one backend, behind one interface.
//
// The engine layer. Everything above it -- the allocator, the
// service, the client's local mode -- runs models through InferenceStrategy
// and never includes a framework header, so adding ONNX Runtime or a vendor
// NPU SDK is one more strategy and nothing else.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "esca/inference/types.hpp"

namespace esca::inference {

struct ConstBuffer {
    const void *data = nullptr;
    size_t size = 0;
};

struct MutableBuffer {
    void *data = nullptr;
    size_t size = 0;
};

class InferenceStrategy {
public:
    virtual ~InferenceStrategy() = default;

    virtual Backend backend() const = 0;

    /// Builds the runtime for this backend. Throws ModelException
    /// (ModelLoadFailed) for a model that cannot be read, RunException
    /// (BackendUnavailable) when this backend cannot run it -- the second is
    /// what lets the allocator fall back to the next backend.
    virtual void load() = 0;
    virtual void unload() = 0;
    virtual bool loaded() const = 0;

    /// Current tensor descriptions; only valid while loaded.
    virtual std::vector<TensorInfo> inputs() const = 0;
    virtual std::vector<TensorInfo> outputs() const = 0;

    /// Gives the inputs these shapes before the next run. A no-op when they
    /// already have them, so callers may pass the shapes every time.
    virtual void resize_inputs(const std::vector<std::vector<int32_t>> &shapes) = 0;

    /// Copies the inputs in, runs, copies the outputs out. Throws
    /// MemoryException (BufferError) when a buffer does not match its tensor.
    virtual void run(const std::vector<ConstBuffer> &inputs,
                     const std::vector<MutableBuffer> &outputs) = 0;

    /// Bytes this strategy holds while loaded, for the memory budget.
    virtual size_t memory_footprint() const = 0;
};

/// Picks the engine from the model file and builds a strategy for one
/// backend.
class InferenceStrategyBuilder {
public:
    /// Whether this build can run anything on the backend at all.
    static bool available(Backend backend);

    /// Throws RunException (BackendUnavailable) for an unknown file type or a
    /// backend this build lacks. Does not load.
    static std::unique_ptr<InferenceStrategy> build(const ModelConfig &config, Backend backend,
                                                     int threads);
};

/// What the allocator builds strategies with: InferenceStrategyBuilder in
/// the service, a fake in the tests.
using StrategyFactory = std::function<std::unique_ptr<InferenceStrategy>(
    const ModelConfig &config, Backend backend, int threads)>;

/// Which backends a factory can serve; paired with StrategyFactory.
using BackendProbe = std::function<bool(Backend)>;

} // namespace esca::inference
