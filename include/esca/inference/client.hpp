// libesca-inference-client: what an application runs a model through.
//
// The inference client. An application builds a Session with
// InferenceBuilder, fills the input buffers it is handed, calls run(), and
// reads the output buffers. Whether the model runs in esca-inferenced --
// buffers in shared memory, control over D-Bus -- or in-process is the
// builder's decision, and the application code is the same either way.
//
//     auto session = esca::inference::InferenceBuilder().model("detect").build();
//     std::memcpy(session->inputs()[0].data, pixels, session->inputs()[0].size);
//     session->run();
//     const float *boxes = session->outputs()[0].as<float>();

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "esca/inference/errors.hpp"
#include "esca/inference/types.hpp"

namespace esca::inference {

/// One input or output, mapped into this process.
struct TensorBuffer {
    TensorInfo info;
    void *data = nullptr;
    size_t size = 0;

    template <typename T>
    T *as() const
    {
        return static_cast<T *>(data);
    }
};

struct InferenceTiming {
    double run_ms = 0;
    double wait_ms = 0;
    Backend backend = Backend::Cpu;
};

/// ErrorCode::Ok on success.
using InferenceCallback =
    std::function<void(ErrorCode code, const std::string &message, const InferenceTiming &timing)>;

class Session {
public:
    virtual ~Session() = default;

    virtual const std::string &model() const = 0;
    virtual std::vector<TensorBuffer> &inputs() = 0;
    /// Shapes are updated after each run, for models with dynamic outputs.
    virtual std::vector<TensorBuffer> &outputs() = 0;

    /// Throws InferenceException.
    virtual InferenceTiming run(int priority = 0) = 0;
    /// Returns at once; `done` runs on another thread. The buffers must not
    /// be touched until it has.
    virtual void run_async(InferenceCallback done, int priority = 0) = 0;

    /// May replace the buffers: re-read inputs() and outputs() afterwards.
    virtual void set_input_shape(size_t index, const std::vector<int32_t> &shape) = 0;

    /// True when the model runs in esca-inferenced.
    virtual bool remote() const = 0;
};

// --- the service, directly -------------------------------------------------------

struct ModelSummaryInfo {
    std::string name;
    std::string state;
    std::string preferred;
    std::string loaded;
    std::string path;
    uint64_t bytes = 0;
    std::string last_error;
};

struct ServiceStats {
    uint32_t queued = 0;
    uint32_t running = 0;
    uint64_t completed = 0;
    uint64_t failed = 0;
    double avg_ms = 0;
    double p95_ms = 0;
    double avg_wait_ms = 0;
    std::string scheduling;
    std::string eviction;
    uint32_t contexts = 0;
    uint64_t loaded_bytes = 0;
};

/// A thin D-Bus client. Every method
/// throws the InferenceException the service raised, or
/// ServiceUnavailable when it is not there.
class InferenceServiceClient {
public:
    explicit InferenceServiceClient(bool session_bus = false);
    ~InferenceServiceClient();
    InferenceServiceClient(const InferenceServiceClient &) = delete;
    InferenceServiceClient &operator=(const InferenceServiceClient &) = delete;

    /// Whether esca-inferenced owns its name on the bus.
    static bool available(bool session_bus = false);

    std::vector<ModelSummaryInfo> list_models();
    std::string model_config(const std::string &name);
    std::pair<std::vector<TensorInfo>, std::vector<TensorInfo>> model_info(const std::string &name);
    std::string register_model(const std::string &config_json);
    void unregister_model(const std::string &name);
    std::string load_model(const std::string &name);
    void unload_model(const std::string &name);
    /// (registered, errors)
    std::pair<std::vector<std::string>, std::vector<std::string>> refresh();
    ServiceStats stats();
    void set_scheduling_policy(const std::string &name);
    void set_eviction_policy(const std::string &name);

    struct Context {
        std::string id;
        std::vector<TensorInfo> inputs;
        std::vector<TensorInfo> outputs;
        std::vector<std::string> input_buffers;
        std::vector<std::string> output_buffers;
    };
    Context create_context(const std::string &model);
    void destroy_context(const std::string &id);
    Context set_input_shape(const std::string &id, uint32_t index, const std::vector<int32_t> &shape);

    /// Waits for the result. `outputs` gets the output shapes as produced.
    InferenceTiming infer(const std::string &id, int priority, std::vector<TensorInfo> *outputs = nullptr,
                          int timeout_ms = 30000);
    /// Returns the request id; `done` runs on the client's bus thread when
    /// the service signals completion.
    uint64_t infer_async(const std::string &id, int priority, InferenceCallback done);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- choosing how to run -----------------------------------------------------------

enum class Mode {
    /// The service if it is on the bus, otherwise in-process.
    Auto,
    Service,
    Local,
};

/// Builds a Session for one model.
class InferenceBuilder {
public:
    /// The model by its registered name (service), or the name of a config
    /// in the service's configuration directory (local).
    InferenceBuilder &model(std::string name);
    /// A model configuration file. Local mode runs it directly; service mode
    /// registers it first if the service does not know the model.
    InferenceBuilder &config(std::string path);
    InferenceBuilder &mode(Mode mode);
    InferenceBuilder &session_bus(bool session);
    /// Where local mode looks up `model(name)` when no config() was given.
    InferenceBuilder &config_dir(std::string dir);

    /// Throws InferenceException when neither way can run the model.
    std::unique_ptr<Session> build();

private:
    std::string model_;
    std::string config_;
    std::string config_dir_ = "/etc/esca/inference/models.d";
    Mode mode_ = Mode::Auto;
    bool session_bus_ = false;
};

} // namespace esca::inference
