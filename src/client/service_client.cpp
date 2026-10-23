// InferenceServiceClient: the service's methods as C++ calls.

#include <chrono>
#include <map>
#include <mutex>

#include "esca/inference/client.hpp"
#include "wire.hpp"

namespace esca::inference {

namespace {

using ContextTuple = std::tuple<std::string, std::vector<wire::Tensor>, std::vector<wire::Tensor>,
                                std::vector<std::string>, std::vector<std::string>>;

InferenceServiceClient::Context context_of(const std::string &id, const std::vector<wire::Tensor> &in,
                                           const std::vector<wire::Tensor> &out,
                                           const std::vector<std::string> &ib,
                                           const std::vector<std::string> &ob)
{
    return { id, wire::from_wire(in), wire::from_wire(out), ib, ob };
}

/// Runs a call, turning D-Bus errors back into the service's exceptions.
template <typename F>
auto call(F &&f) -> decltype(f())
{
    try {
        return f();
    } catch (const sdbus::Error &e) {
        wire::rethrow(e);
    }
}

} // namespace

struct InferenceServiceClient::Impl {
    std::unique_ptr<sdbus::IProxy> proxy;

    std::mutex mutex;
    std::map<uint64_t, InferenceCallback> waiting;
    /// Results that arrived before InferAsync returned their id.
    struct Early {
        ErrorCode code;
        std::string message;
        InferenceTiming timing;
    };
    std::map<uint64_t, Early> early;

    void completed(uint64_t request, const std::string &error, const std::string &message, double run,
                   double wait, const std::string &backend)
    {
        InferenceTiming timing { run, wait, backend_from_string(backend).value_or(Backend::Cpu) };
        const ErrorCode code = error.empty() ? ErrorCode::Ok : error_code_from_string(error);
        InferenceCallback done;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto it = waiting.find(request);
            if (it == waiting.end()) {
                early[request] = Early { code, message, timing };
                return;
            }
            done = std::move(it->second);
            waiting.erase(it);
        }
        done(code, message, timing);
    }
};

InferenceServiceClient::InferenceServiceClient(bool session_bus) : impl_(std::make_unique<Impl>())
{
    try {
        auto connection =
            session_bus ? sdbus::createSessionBusConnection() : sdbus::createSystemBusConnection();
        // The proxy owns the connection and runs its event loop on a thread
        // of its own, which is what delivers InferenceCompleted.
        impl_->proxy = sdbus::createProxy(std::move(connection), sdbus::ServiceName { wire::kService },
                                          sdbus::ObjectPath { wire::kPath });
        impl_->proxy->uponSignal("InferenceCompleted")
            .onInterface(wire::kInterface)
            .call([impl = impl_.get()](const std::string &, uint64_t request, const std::string &error,
                                       const std::string &message, double run, double wait,
                                       const std::string &backend) {
                impl->completed(request, error, message, run, wait, backend);
            });
    } catch (const sdbus::Error &e) {
        raise(ErrorCode::ServiceUnavailable, std::string("cannot reach the bus: ") + e.getMessage());
    }
}

InferenceServiceClient::~InferenceServiceClient() = default;

bool InferenceServiceClient::available(bool session_bus)
{
    try {
        auto connection =
            session_bus ? sdbus::createSessionBusConnection() : sdbus::createSystemBusConnection();
        auto bus = sdbus::createProxy(*connection, sdbus::ServiceName { "org.freedesktop.DBus" },
                                      sdbus::ObjectPath { "/org/freedesktop/DBus" });
        bool owned = false;
        bus->callMethod("NameHasOwner")
            .onInterface("org.freedesktop.DBus")
            .withArguments(std::string(wire::kService))
            .storeResultsTo(owned);
        return owned;
    } catch (const sdbus::Error &) {
        return false;
    }
}

std::vector<ModelSummaryInfo> InferenceServiceClient::list_models()
{
    return call([&] {
        std::vector<wire::ModelRow> rows;
        impl_->proxy->callMethod("ListModels").onInterface(wire::kInterface).storeResultsTo(rows);
        std::vector<ModelSummaryInfo> out;
        for (const auto &r : rows)
            out.push_back({ r.get<0>(), r.get<1>(), r.get<2>(), r.get<3>(), r.get<4>(), r.get<5>(), r.get<6>() });
        return out;
    });
}

std::string InferenceServiceClient::model_config(const std::string &name)
{
    return call([&] {
        std::string json;
        impl_->proxy->callMethod("GetModelConfig").onInterface(wire::kInterface).withArguments(name).storeResultsTo(json);
        return json;
    });
}

std::pair<std::vector<TensorInfo>, std::vector<TensorInfo>> InferenceServiceClient::model_info(const std::string &name)
{
    return call([&] {
        std::vector<wire::Tensor> in, out;
        impl_->proxy->callMethod("GetModelInfo")
            .onInterface(wire::kInterface)
            .withTimeout(std::chrono::seconds(60))
            .withArguments(name)
            .storeResultsTo(in, out);
        return std::make_pair(wire::from_wire(in), wire::from_wire(out));
    });
}

std::string InferenceServiceClient::register_model(const std::string &config_json)
{
    return call([&] {
        std::string name;
        impl_->proxy->callMethod("RegisterModel").onInterface(wire::kInterface).withArguments(config_json).storeResultsTo(name);
        return name;
    });
}

void InferenceServiceClient::unregister_model(const std::string &name)
{
    call([&] { impl_->proxy->callMethod("UnregisterModel").onInterface(wire::kInterface).withArguments(name); });
}

std::string InferenceServiceClient::load_model(const std::string &name)
{
    return call([&] {
        std::string backend;
        impl_->proxy->callMethod("LoadModel")
            .onInterface(wire::kInterface)
            .withTimeout(std::chrono::seconds(60))
            .withArguments(name)
            .storeResultsTo(backend);
        return backend;
    });
}

void InferenceServiceClient::unload_model(const std::string &name)
{
    call([&] { impl_->proxy->callMethod("UnloadModel").onInterface(wire::kInterface).withArguments(name); });
}

std::pair<std::vector<std::string>, std::vector<std::string>> InferenceServiceClient::refresh()
{
    return call([&] {
        std::vector<std::string> registered, errors;
        impl_->proxy->callMethod("Refresh").onInterface(wire::kInterface).storeResultsTo(registered, errors);
        return std::make_pair(registered, errors);
    });
}

ServiceStats InferenceServiceClient::stats()
{
    return call([&] {
        sdbus::Struct<uint32_t, uint32_t, uint64_t, uint64_t, double, double, double, std::string, std::string,
                      uint32_t, uint64_t>
            s;
        impl_->proxy->callMethod("GetStats").onInterface(wire::kInterface).storeResultsTo(s);
        return ServiceStats { s.get<0>(), s.get<1>(), s.get<2>(), s.get<3>(), s.get<4>(), s.get<5>(),
                              s.get<6>(), s.get<7>(), s.get<8>(), s.get<9>(), s.get<10>() };
    });
}

void InferenceServiceClient::set_scheduling_policy(const std::string &name)
{
    call([&] { impl_->proxy->callMethod("SetSchedulingPolicy").onInterface(wire::kInterface).withArguments(name); });
}

void InferenceServiceClient::set_eviction_policy(const std::string &name)
{
    call([&] { impl_->proxy->callMethod("SetEvictionPolicy").onInterface(wire::kInterface).withArguments(name); });
}

InferenceServiceClient::Context InferenceServiceClient::create_context(const std::string &model)
{
    return call([&] {
        std::string id;
        std::vector<wire::Tensor> in, out;
        std::vector<std::string> ib, ob;
        impl_->proxy->callMethod("CreateContext")
            .onInterface(wire::kInterface)
            .withTimeout(std::chrono::seconds(60)) // may load the model first
            .withArguments(model)
            .storeResultsTo(id, in, out, ib, ob);
        return context_of(id, in, out, ib, ob);
    });
}

void InferenceServiceClient::destroy_context(const std::string &id)
{
    call([&] { impl_->proxy->callMethod("DestroyContext").onInterface(wire::kInterface).withArguments(id); });
}

InferenceServiceClient::Context InferenceServiceClient::set_input_shape(const std::string &id, uint32_t index,
                                                                        const std::vector<int32_t> &shape)
{
    return call([&] {
        std::string cid;
        std::vector<wire::Tensor> in, out;
        std::vector<std::string> ib, ob;
        impl_->proxy->callMethod("SetInputShape")
            .onInterface(wire::kInterface)
            .withTimeout(std::chrono::seconds(60))
            .withArguments(id, index, shape)
            .storeResultsTo(cid, in, out, ib, ob);
        return context_of(cid, in, out, ib, ob);
    });
}

InferenceTiming InferenceServiceClient::infer(const std::string &id, int priority, std::vector<TensorInfo> *outputs,
                                              int timeout_ms)
{
    return call([&] {
        double run = 0, wait = 0;
        std::string backend;
        std::vector<wire::Tensor> out;
        impl_->proxy->callMethod("Infer")
            .onInterface(wire::kInterface)
            .withTimeout(std::chrono::milliseconds(timeout_ms))
            .withArguments(id, static_cast<int32_t>(priority))
            .storeResultsTo(run, wait, backend, out);
        if (outputs)
            *outputs = wire::from_wire(out);
        return InferenceTiming { run, wait, backend_from_string(backend).value_or(Backend::Cpu) };
    });
}

uint64_t InferenceServiceClient::infer_async(const std::string &id, int priority, InferenceCallback done)
{
    const uint64_t request = call([&] {
        uint64_t r = 0;
        impl_->proxy->callMethod("InferAsync")
            .onInterface(wire::kInterface)
            .withArguments(id, static_cast<int32_t>(priority))
            .storeResultsTo(r);
        return r;
    });

    Impl::Early early;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto it = impl_->early.find(request);
        if (it == impl_->early.end()) {
            impl_->waiting[request] = std::move(done);
            return request;
        }
        early = it->second;
        impl_->early.erase(it);
    }
    done(early.code, early.message, early.timing);
    return request;
}

} // namespace esca::inference
