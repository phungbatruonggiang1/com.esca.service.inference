#include "service.hpp"

#include <future>
#include <thread>

#include <nlohmann/json.hpp>

#include "esca/inference/errors.hpp"
#include "logger.hpp"
#include "model_config.hpp"
#include "wire.hpp"

namespace esca::inference {

namespace {

constexpr const char *kCapabilityService = "vn.sparc.esca.Capability1";

using wire::Tensor;
using ContextReply = std::tuple<std::string, std::vector<Tensor>, std::vector<Tensor>,
                                std::vector<std::string>, std::vector<std::string>>;

ContextReply reply_of(const ContextView &v)
{
    return { v.id, wire::to_wire(v.inputs), wire::to_wire(v.outputs), v.input_buffers, v.output_buffers };
}

std::string join(const std::vector<Backend> &backends)
{
    std::string out;
    for (Backend b : backends)
        out += (out.empty() ? "" : ",") + std::string(to_string(b));
    return out;
}

/// Runs a handler on the bus thread, turning failures into typed D-Bus
/// errors instead of the generic one sdbus would make of an exception.
template <typename F>
auto guarded(F &&f) -> decltype(f())
{
    try {
        return f();
    } catch (const InferenceException &e) {
        throw wire::to_dbus(e.code(), e.what());
    } catch (const sdbus::Error &) {
        throw;
    } catch (const std::exception &e) {
        throw wire::to_dbus(ErrorCode::Internal, e.what());
    }
}

/// A method's pending reply, shared between the handler and the thread that
/// answers it: sdbus::Result is move-only and std::function copies.
template <typename R>
std::shared_ptr<R> share(R &&result)
{
    return std::make_shared<R>(std::move(result));
}

/// The same for work answered later from another thread.
template <typename R, typename F>
void answer(const std::shared_ptr<R> &result, F &&f)
{
    try {
        f();
    } catch (const InferenceException &e) {
        result->returnError(wire::to_dbus(e.code(), e.what()));
    } catch (const std::exception &e) {
        result->returnError(wire::to_dbus(ErrorCode::Internal, e.what()));
    }
}

ModelManagerOptions manager_options(const ServiceConfig &c)
{
    ModelManagerOptions o;
    o.memory_reserve_bytes = c.memory_reserve_mb << 20;
    o.pressure_threshold = c.pressure_threshold;
    o.cores = std::max(1u, std::thread::hardware_concurrency());
    o.cpu_slots = c.cpu_slots;
    return o;
}

} // namespace

InferenceService::InferenceService(sdbus::IConnection &connection, const ServiceConfig &config)
    : connection_(connection), config_(config),
      models_(InferenceStrategyBuilder::build, InferenceStrategyBuilder::available,
              make_eviction_policy(config.eviction), std::make_shared<MemoryMonitor>(),
              manager_options(config)),
      buffers_(config.shm_group),
      scheduler_(std::make_unique<InferenceScheduler>(models_, make_scheduling_policy(config.scheduling),
                                                      config.workers)),
      contexts_(std::make_unique<ContextManager>(models_, *scheduler_, buffers_)),
      registration_(models_), pool_(std::make_unique<ThreadPool>(config.pool_threads))
{
    object_ = sdbus::createObject(connection_, sdbus::ObjectPath { config_.object_path });

    object_->addVTable(
        // --- models ------------------------------------------------------------
        sdbus::registerMethod("ListModels").implementedAs([this] {
            return guarded([&] {
                std::vector<wire::ModelRow> rows;
                for (const ModelSummary &m : models_.list())
                    rows.emplace_back(m.name, std::string(to_string(m.state)), join(m.preferred),
                                      join(m.loaded), m.model_path, static_cast<uint64_t>(m.bytes),
                                      m.last_error);
                return rows;
            });
        }),
        sdbus::registerMethod("GetModelConfig").implementedAs([this](const std::string &name) {
            return guarded([&] { return to_json(models_.config(name)); });
        }),
        sdbus::registerMethod("GetModelInfo")
            .implementedAs([this](sdbus::Result<std::vector<Tensor>, std::vector<Tensor>> &&r,
                                  std::string name) {
                auto result = share(std::move(r));
                pool_->post([this, result, name] {
                    answer(result, [&] {
                        const ModelMetadata m = models_.metadata(name);
                        result->returnResults(wire::to_wire(m.inputs), wire::to_wire(m.outputs));
                    });
                });
            }),
        sdbus::registerMethod("RegisterModel").implementedAs([this](const std::string &json) {
            return guarded([&] {
                const std::string name = registration_.register_json(json);
                publish_capability();
                return name;
            });
        }),
        sdbus::registerMethod("UnregisterModel").implementedAs([this](const std::string &name) {
            guarded([&] { models_.unregister_model(name); });
            publish_capability();
        }),
        sdbus::registerMethod("LoadModel")
            .implementedAs([this](sdbus::Result<std::string> &&r, std::string name) {
                auto result = share(std::move(r));
                pool_->post([this, result, name] {
                    answer(result, [&] { result->returnResults(std::string(to_string(models_.load(name)))); });
                });
            }),
        sdbus::registerMethod("UnloadModel").implementedAs([this](const std::string &name) {
            guarded([&] { models_.unload(name); });
        }),
        sdbus::registerMethod("Refresh").implementedAs([this] {
            return guarded([&] {
                const RegistrationReport r = registration_.scan(config_.config_dir);
                publish_capability();
                return std::make_tuple(r.registered, r.errors);
            });
        }),

        // --- contexts ------------------------------------------------------------
        sdbus::registerMethod("CreateContext")
            .implementedAs([this](sdbus::Result<std::string, std::vector<Tensor>, std::vector<Tensor>,
                                                std::vector<std::string>, std::vector<std::string>> &&r,
                                  std::string model) {
                auto result = share(std::move(r));
                const std::string owner = sender();
                pool_->post([this, result, model, owner] {
                    answer(result, [&] {
                        const auto [id, in, out, ib, ob] = reply_of(contexts_->create(model, owner));
                        result->returnResults(id, in, out, ib, ob);
                    });
                });
            }),
        sdbus::registerMethod("CreateContextAsync").implementedAs([this](const std::string &model) {
            const uint64_t token = next_token_++;
            const std::string owner = sender();
            pool_->post([this, token, model, owner] {
                std::string id, error, message;
                try {
                    id = contexts_->create(model, owner).id;
                } catch (const InferenceException &e) {
                    error = dbus_error_name(e.code());
                    message = e.what();
                }
                if (!stopping_)
                    object_->emitSignal("ContextCreated")
                        .onInterface(config_.service_name)
                        .withArguments(token, id, error, message);
            });
            return token;
        }),
        sdbus::registerMethod("GetContext").implementedAs([this](const std::string &id) {
            return guarded([&] { return reply_of(contexts_->view(id, sender())); });
        }),
        sdbus::registerMethod("DestroyContext").implementedAs([this](const std::string &id) {
            guarded([&] { contexts_->destroy(id, sender()); });
        }),
        sdbus::registerMethod("SetInputShape")
            .implementedAs([this](sdbus::Result<std::string, std::vector<Tensor>, std::vector<Tensor>,
                                                std::vector<std::string>, std::vector<std::string>> &&r,
                                  std::string id, uint32_t index, std::vector<int32_t> shape) {
                auto result = share(std::move(r));
                const std::string owner = sender();
                pool_->post([this, result, id, index, shape, owner] {
                    answer(result, [&] {
                        const auto [cid, in, out, ib, ob] =
                            reply_of(contexts_->set_input_shape(id, owner, index, shape));
                        result->returnResults(cid, in, out, ib, ob);
                    });
                });
            }),

        // --- inference -------------------------------------------------------------
        sdbus::registerMethod("Infer")
            .implementedAs([this](sdbus::Result<double, double, std::string, std::vector<Tensor>> &&r,
                                  std::string id, int32_t priority) {
                auto result = share(std::move(r));
                try {
                    contexts_->infer(id, sender(), priority, [result](const InferenceResult &res) {
                        if (res.code != ErrorCode::Ok)
                            result->returnError(wire::to_dbus(res.code, res.message));
                        else
                            result->returnResults(res.run_ms, res.wait_ms, std::string(to_string(res.backend)),
                                                  wire::to_wire(res.outputs));
                    });
                } catch (const InferenceException &e) {
                    result->returnError(wire::to_dbus(e.code(), e.what()));
                }
            }),
        sdbus::registerMethod("InferAsync").implementedAs([this](const std::string &id, int32_t priority) {
            return guarded([&] {
                auto request = std::make_shared<uint64_t>(0);
                // The id is known only after submit(); a request finishing
                // before it is stored would report 0, so wait for it.
                auto ready = std::make_shared<std::promise<void>>();
                std::shared_future<void> stored = ready->get_future().share();
                *request = contexts_->infer(id, sender(), priority,
                                            [this, id, request, stored](const InferenceResult &res) {
                                                stored.wait();
                                                if (stopping_)
                                                    return;
                                                object_->emitSignal("InferenceCompleted")
                                                    .onInterface(config_.service_name)
                                                    .withArguments(id, *request,
                                                                   res.code == ErrorCode::Ok
                                                                       ? std::string()
                                                                       : dbus_error_name(res.code),
                                                                   res.message, res.run_ms, res.wait_ms,
                                                                   std::string(to_string(res.backend)));
                                            });
                ready->set_value();
                return *request;
            });
        }),

        // --- service -----------------------------------------------------------------
        sdbus::registerMethod("GetStats").implementedAs([this] {
            const SchedulerStats s = scheduler_->stats();
            return sdbus::Struct<uint32_t, uint32_t, uint64_t, uint64_t, double, double, double, std::string,
                                 std::string, uint32_t, uint64_t>(
                static_cast<uint32_t>(s.queued), static_cast<uint32_t>(s.running), s.completed, s.failed,
                s.avg_ms, s.p95_ms, s.avg_wait_ms, s.policy, models_.eviction_policy(),
                static_cast<uint32_t>(contexts_->count()), static_cast<uint64_t>(models_.loaded_bytes()));
        }),
        sdbus::registerMethod("SetSchedulingPolicy").implementedAs([this](const std::string &name) {
            guarded([&] {
                auto policy = make_scheduling_policy(name);
                if (!policy)
                    raise(ErrorCode::InvalidArgument, "unknown scheduling policy \"" + name + "\" (fifo, priority)");
                scheduler_->set_policy(std::move(policy));
            });
        }),
        sdbus::registerMethod("SetEvictionPolicy").implementedAs([this](const std::string &name) {
            guarded([&] {
                auto policy = make_eviction_policy(name);
                if (!policy)
                    raise(ErrorCode::InvalidArgument, "unknown eviction policy \"" + name + "\" (lru, none)");
                models_.set_eviction_policy(std::move(policy));
            });
        }),

        sdbus::registerSignal("ModelStateChanged").withParameters<std::string, std::string, std::string>(),
        sdbus::registerSignal("ContextCreated").withParameters<uint64_t, std::string, std::string, std::string>(),
        sdbus::registerSignal("InferenceCompleted")
            .withParameters<std::string, uint64_t, std::string, std::string, double, double, std::string>())
        .forInterface(config_.service_name);

    models_.set_listener([this](const std::string &name, ModelState state, const std::string &backend) {
        if (stopping_)
            return;
        object_->emitSignal("ModelStateChanged")
            .onInterface(config_.service_name)
            .withArguments(name, std::string(to_string(state)), backend);
    });

    // A client that exits without destroying its contexts would otherwise
    // leave their buffers in /dev/shm until the service restarts.
    bus_ = sdbus::createProxy(connection_, sdbus::ServiceName { "org.freedesktop.DBus" },
                              sdbus::ObjectPath { "/org/freedesktop/DBus" });
    bus_->uponSignal("NameOwnerChanged")
        .onInterface("org.freedesktop.DBus")
        .call([this](const std::string &name, const std::string &, const std::string &new_owner) {
            if (new_owner.empty() && !name.empty() && name[0] == ':')
                on_client_gone(name);
            else if (name == kCapabilityService && !new_owner.empty())
                publish_capability(); // the provider (re)started
        });
    capability_ = sdbus::createProxy(connection_, sdbus::ServiceName { kCapabilityService },
                                     sdbus::ObjectPath { "/vn/sparc/esca/Capability1" });
}

InferenceService::~InferenceService()
{
    // Order matters: no more signals, then the jobs that might still use the
    // contexts, then the workers that might still answer requests.
    stopping_ = true;
    pool_.reset();
    scheduler_.reset();
    contexts_.reset();
}

std::string InferenceService::sender() const
{
    const char *s = object_->getCurrentlyProcessedMessage().getSender();
    return s ? s : "";
}

void InferenceService::on_client_gone(const std::string &name)
{
    const size_t n = contexts_->destroy_owned_by(name);
    if (n > 0)
        ESCA_LOG_INFO("%s left: destroyed %zu context(s)", name.c_str(), n);
}

void InferenceService::start()
{
    const RegistrationReport r = registration_.scan(config_.config_dir);
    ESCA_LOG_INFO("%zu model(s) registered from %s, %zu skipped", r.registered.size(),
                  config_.config_dir.c_str(), r.errors.size());
    publish_capability();
}

void InferenceService::publish_capability()
{
    nlohmann::json backends = nlohmann::json::array();
    for (Backend b : { Backend::Npu, Backend::Gpu, Backend::Xnnpack, Backend::Cpu })
        if (InferenceStrategyBuilder::available(b))
            backends.push_back(to_string(b));
    nlohmann::json tasks = nlohmann::json::array(), models = nlohmann::json::array();
    for (const ModelSummary &m : models_.list()) {
        tasks.push_back(m.name);
        models.push_back({ { "name", m.name }, { "backends", join(m.preferred) } });
    }
    const nlohmann::json ai = { { "service", config_.service_name },
                                { "engine", "tflite" },
                                { "backends", backends },
                                { "tasks", tasks },
                                { "models", models } };
    // Asynchronous: this may run on the bus thread, inside another call.
    // Without a provider on the bus the call just fails, which is fine.
    capability_->callMethodAsync("RegisterManifestInfo")
        .onInterface(kCapabilityService)
        .withArguments(std::string("ai"), ai.dump())
        .uponReplyInvoke([](std::optional<sdbus::Error> error, uint64_t version) {
            if (error)
                ESCA_LOG_DEBUG("capability provider: %s", error->getMessage().c_str());
            else
                ESCA_LOG_INFO("capabilities published, manifest version %llu",
                              static_cast<unsigned long long>(version));
        });
}

} // namespace esca::inference
