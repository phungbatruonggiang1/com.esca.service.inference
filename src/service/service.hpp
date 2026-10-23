// esca-inferenced: the D-Bus face of the inference stack.
//
// The inference service. Exposes models, contexts and inference on
// vn.sparc.esca.Inference1, delegates the work to the context manager, the
// registration helper and the thread pool, emits completion signals for
// asynchronous calls, and turns every failure into a typed D-Bus error.
//
// Nothing that may take long -- loading a model, running one -- happens on
// the bus thread: such calls are answered later through sdbus::Result, so a
// slow model never stalls another client's ListModels().

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include <sdbus-c++/sdbus-c++.h>

#include "buffer_allocator.hpp"
#include "config_manager.hpp"
#include "context_manager.hpp"
#include "model_manager.hpp"
#include "model_registration.hpp"
#include "scheduler.hpp"
#include "thread_pool.hpp"

namespace esca::inference {

class InferenceService {
public:
    InferenceService(sdbus::IConnection &connection, const ServiceConfig &config);
    ~InferenceService();

    /// Registers what the configuration directory holds.
    void start();

private:
    std::string sender() const;
    void on_client_gone(const std::string &name);
    /// Tells the capability provider, if it runs, what this device can run:
    /// the "ai" section of its manifest, so peers deciding where to offload
    /// see the models and backends here.
    void publish_capability();

    sdbus::IConnection &connection_;
    ServiceConfig config_;
    std::atomic<bool> stopping_ { false };
    std::atomic<uint64_t> next_token_ { 1 };

    ModelManager models_;
    BufferAllocator buffers_;
    std::unique_ptr<InferenceScheduler> scheduler_;
    std::unique_ptr<ContextManager> contexts_;
    ModelRegistrationHelper registration_;
    std::unique_ptr<ThreadPool> pool_;

    std::unique_ptr<sdbus::IObject> object_;
    std::unique_ptr<sdbus::IProxy> bus_;
    std::unique_ptr<sdbus::IProxy> capability_;
};

} // namespace esca::inference
