// Inference contexts: one client's use of one model.
//
// The context manager. A context owns a shared-memory buffer per input
// and output, the input shapes its client set, and the bus name of the
// client that made it -- so a client that dies takes its contexts with it
// instead of leaking buffers.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "buffer_allocator.hpp"
#include "model_manager.hpp"
#include "scheduler.hpp"

namespace esca::inference {

/// What a client is told about its context: enough to map every buffer.
struct ContextView {
    std::string id;
    std::string model;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    std::vector<std::string> input_buffers;  ///< shm names, one per input
    std::vector<std::string> output_buffers; ///< shm names, one per output
};

class ContextManager {
public:
    ContextManager(ModelManager &models, InferenceScheduler &scheduler, BufferAllocator &buffers);

    /// Loads the model if needed, then allocates its buffers. Throws
    /// ModelNotFound, ModelLoadFailed, OutOfMemory, BufferError.
    ContextView create(const std::string &model, const std::string &owner);

    /// Throws ContextNotFound, or InvalidArgument when `owner` did not make
    /// it. An empty owner is the service itself and may destroy any.
    void destroy(const std::string &id, const std::string &owner);

    /// Every context `owner` made; returns how many.
    size_t destroy_owned_by(const std::string &owner);

    ContextView view(const std::string &id, const std::string &owner) const;

    /// New shape for one input. Buffers whose size changes are replaced,
    /// under new names; the view says which to map.
    ContextView set_input_shape(const std::string &id, const std::string &owner, size_t index,
                                const std::vector<int32_t> &shape);

    /// Queues an inference on the context's buffers; `done` runs on a worker
    /// thread. Throws ContextNotFound, or ModelBusy while the previous
    /// inference on this context has not finished -- one at a time keeps
    /// its output buffers from being overwritten while the client reads them.
    uint64_t infer(const std::string &id, const std::string &owner, int priority,
                   std::function<void(const InferenceResult &)> done);

    size_t count() const;

private:
    struct Context {
        std::string id;
        std::string model;
        std::string owner;
        std::vector<std::vector<int32_t>> shapes;
        std::vector<TensorInfo> inputs;
        std::vector<TensorInfo> outputs;
        std::vector<std::shared_ptr<SharedBuffer>> input_buffers;
        std::vector<std::shared_ptr<SharedBuffer>> output_buffers;
        std::atomic<bool> busy { false };
        unsigned generation = 0; ///< bumped when buffers are replaced
        std::mutex mutex;        ///< guards everything but busy
    };

    std::shared_ptr<Context> find(const std::string &id, const std::string &owner) const;
    void allocate(Context &context, const ModelMetadata &metadata);
    static ContextView view_of(const Context &context);

    ModelManager &models_;
    InferenceScheduler &scheduler_;
    BufferAllocator &buffers_;

    mutable std::mutex mutex_;
    std::map<std::string, std::shared_ptr<Context>> contexts_;
    std::atomic<uint64_t> next_id_ { 1 };
};

} // namespace esca::inference
