// The resource allocator's decisions, against fake models and fake memory:
// lazy loading, backend fallback, per-request device selection, eviction
// under memory pressure, and scheduling order.

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

#include "check.hpp"
#include "fakes.hpp"
#include "model_config.hpp"
#include "model_manager.hpp"
#include "scheduler.hpp"

using namespace esca::inference;
using namespace std::chrono_literals;

namespace {

constexpr size_t MiB = 1u << 20;

/// A model file of `mib` MiB, sparse, so estimate_load_bytes has a size.
std::string model_file(const std::string &name, size_t mib)
{
    const std::string path = "/tmp/esca-inference-test-" + name + ".tflite";
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd >= 0) {
        if (::ftruncate(fd, static_cast<off_t>(mib * MiB)) != 0)
            std::perror("ftruncate");
        ::close(fd);
    }
    return path;
}

ModelConfig config(const std::string &name, std::vector<Backend> backends = { Backend::Xnnpack, Backend::Cpu },
                   size_t mib = 1)
{
    ModelConfig c;
    c.name = name;
    c.model_path = model_file(name, mib);
    c.backends = std::move(backends);
    return c;
}

struct Rig {
    std::shared_ptr<fake::Memory> memory = std::make_shared<fake::Memory>();
    std::unique_ptr<ModelManager> models;

    explicit Rig(const char *eviction = "lru", size_t reserve_mib = 128)
    {
        fake::reset();
        ModelManagerOptions o;
        o.memory_reserve_bytes = reserve_mib * MiB;
        o.watch_period = 1h; // the tests call relieve_pressure() themselves
        o.cpu_slots = 1;
        models = std::make_unique<ModelManager>(fake::factory(), fake::probe(),
                                                make_eviction_policy(eviction), memory, o);
    }
};

/// Runs `name` once and returns the doubled first element.
float run_once(ModelManager &models, const std::string &name, Backend &used)
{
    float in[4] = { 1, 2, 3, 4 }, out[4] = {};
    models.acquire(name);
    models.run(name, {}, { { in, sizeof(in) } }, { { out, sizeof(out) } }, used);
    models.release(name);
    return out[0];
}

} // namespace

TEST("register, list, and refuse a duplicate")
{
    Rig r;
    r.models->register_model(config("a"));
    CHECK(r.models->contains("a"));
    CHECK(r.models->list().size() == 1);
    CHECK(r.models->list()[0].state == ModelState::Registered);
    CHECK_THROWS_CODE(r.models->register_model(config("a")), ErrorCode::ModelAlreadyRegistered);
    CHECK_THROWS_CODE(r.models->unload("nope"), ErrorCode::ModelNotFound);
}

TEST("FR03 a model loads on its first request, on its preferred backend")
{
    Rig r;
    r.models->register_model(config("a"));
    CHECK(fake::world().loads == 0);
    Backend used = Backend::Npu;
    CHECK(run_once(*r.models, "a", used) == 2.0f);
    CHECK(used == Backend::Xnnpack);
    CHECK(fake::world().loads == 1);
    CHECK(r.models->list()[0].state == ModelState::Loaded);
}

TEST("FR04 a backend that cannot run the model falls back to the next")
{
    Rig r;
    fake::world().refuse = { Backend::Xnnpack };
    r.models->register_model(config("a"));
    Backend used = Backend::Npu;
    CHECK(run_once(*r.models, "a", used) == 2.0f);
    CHECK(used == Backend::Cpu);
    // And it is not tried again on the next request.
    run_once(*r.models, "a", used);
    CHECK(used == Backend::Cpu);
}

TEST("FR04 backends this machine lacks are never tried")
{
    Rig r;
    r.models->register_model(config("a", { Backend::Npu, Backend::Gpu, Backend::Cpu }));
    Backend used = Backend::Npu;
    run_once(*r.models, "a", used);
    CHECK(used == Backend::Cpu);
    CHECK(fake::world().loads == 1);
}

TEST("QA02 a busy accelerator sends the request to the next free backend")
{
    Rig r;
    fake::world().available = { Backend::Npu, Backend::Cpu };
    r.models->register_model(config("a", { Backend::Npu, Backend::Cpu }));
    Backend used = Backend::Cpu;
    run_once(*r.models, "a", used);
    CHECK(used == Backend::Npu);

    // Another model holds the NPU.
    const_cast<DeviceSelector &>(r.models->selector()).acquire(Backend::Npu);
    run_once(*r.models, "a", used);
    CHECK(used == Backend::Cpu);
    const_cast<DeviceSelector &>(r.models->selector()).release(Backend::Npu);

    run_once(*r.models, "a", used);
    CHECK(used == Backend::Npu);
}

TEST("QA03 a load that would breach the reserve evicts the least recently used idle model")
{
    Rig r("lru", 100);
    r.models->register_model(config("old", { Backend::Cpu }, 20));
    r.models->register_model(config("new", { Backend::Cpu }, 20));
    // Each loaded fake holds 30 MiB; 160 MiB to start with.
    fake::world().footprint = 30 * MiB;
    r.memory->capacity = 160 * MiB;
    Backend used;
    run_once(*r.models, "old", used);

    // 130 MiB left; "new" wants ~40 for itself plus 100 in reserve. Only
    // unloading "old" makes room.
    run_once(*r.models, "new", used);
    for (const ModelSummary &s : r.models->list()) {
        if (s.name == "old")
            CHECK(s.loaded.empty());
        if (s.name == "new")
            CHECK(!s.loaded.empty());
    }
}

TEST("QA03 without an eviction policy, a load that does not fit says OutOfMemory")
{
    Rig r("none", 100);
    r.models->register_model(config("a", { Backend::Cpu }, 20));
    r.memory->capacity = 50 * MiB;
    Backend used;
    float in[4] = {}, out[4] = {};
    r.models->acquire("a");
    CHECK_THROWS_CODE(r.models->run("a", {}, { { in, 16 } }, { { out, 16 } }, used), ErrorCode::OutOfMemory);
    r.models->release("a");
}

TEST("QA03 under pressure an idle model is unloaded, a busy one is not")
{
    Rig r("lru", 100);
    r.models->register_model(config("busy", { Backend::Cpu }));
    r.models->register_model(config("idle", { Backend::Cpu }));
    Backend used;
    run_once(*r.models, "idle", used);
    run_once(*r.models, "busy", used);
    r.models->acquire("busy");

    // Two fakes at 10 MiB each: 70 - 20 leaves 50, below the 100 reserve.
    r.memory->capacity = 70 * MiB;
    r.models->relieve_pressure();
    r.models->relieve_pressure();
    for (const ModelSummary &s : r.models->list()) {
        if (s.name == "idle")
            CHECK(s.loaded.empty());
        if (s.name == "busy")
            CHECK(!s.loaded.empty());
    }
    CHECK_THROWS_CODE(r.models->unload("busy"), ErrorCode::ModelBusy);
    r.models->release("busy");
    r.models->unload("busy");
}

TEST("metadata loads the model if it never was")
{
    Rig r;
    r.models->register_model(config("a"));
    const ModelMetadata m = r.models->metadata("a");
    CHECK(m.inputs.size() == 1 && m.inputs[0].shape == std::vector<int32_t>({ 1, 4 }));
    CHECK(m.outputs[0].bytes == 16);
}

TEST("scheduler: FIFO runs in arrival order")
{
    Rig r;
    for (const char *n : { "m1", "m2", "m3" })
        r.models->register_model(config(n, { Backend::Cpu }));
    fake::world().run_delay = 20ms;
    InferenceScheduler s(*r.models, make_scheduling_policy("fifo"), 1);
    float in[4] = {}, out[3][4];
    std::atomic<int> done { 0 };
    for (int i = 0; i < 3; ++i) {
        InferenceRequest q;
        q.model = "m" + std::to_string(i + 1);
        q.inputs = { { in, 16 } };
        q.outputs = { { out[i], 16 } };
        q.done = [&](const InferenceResult &) { ++done; };
        s.submit(std::move(q));
    }
    while (done < 3)
        std::this_thread::sleep_for(5ms);
    CHECK(fake::world().order == std::vector<std::string>({ "m1", "m2", "m3" }));
    CHECK(s.stats().completed == 3);
    CHECK(s.stats().failed == 0);
}

TEST("scheduler: the priority policy takes the most urgent waiting request")
{
    Rig r;
    for (const char *n : { "block", "low", "high" })
        r.models->register_model(config(n, { Backend::Cpu }));
    fake::world().run_delay = 30ms;
    InferenceScheduler s(*r.models, make_scheduling_policy("priority"), 1);
    float in[4] = {}, out[3][4];
    std::atomic<int> done { 0 };
    auto submit = [&](const char *model, int priority, int slot) {
        InferenceRequest q;
        q.model = model;
        q.priority = priority;
        q.inputs = { { in, 16 } };
        q.outputs = { { out[slot], 16 } };
        q.done = [&](const InferenceResult &) { ++done; };
        s.submit(std::move(q));
    };
    submit("block", 0, 0); // occupies the only worker
    std::this_thread::sleep_for(5ms);
    submit("low", 0, 1);
    submit("high", 5, 2);
    while (done < 3)
        std::this_thread::sleep_for(5ms);
    CHECK(fake::world().order == std::vector<std::string>({ "block", "high", "low" }));
}

TEST("scheduler: a failure is reported and counted, not thrown")
{
    Rig r;
    fake::world().refuse = { Backend::Cpu, Backend::Xnnpack };
    r.models->register_model(config("a"));
    InferenceScheduler s(*r.models, nullptr, 1);
    float in[4] = {}, out[4];
    InferenceRequest q;
    q.model = "a";
    q.inputs = { { in, 16 } };
    q.outputs = { { out, 16 } };
    const InferenceResult result = s.run(std::move(q));
    CHECK(result.code == ErrorCode::BackendUnavailable);
    CHECK(s.stats().failed == 1);
    InferenceRequest missing;
    missing.model = "missing";
    CHECK_THROWS_CODE(s.submit(std::move(missing)), ErrorCode::ModelNotFound);
}

TEST("config: all fields parse, and the name must match the file")
{
    const ModelConfig c = parse_model_config(R"({
        "name": "scrfd_quantized",
        "modelPath": "/data/models/face/scrfd_quantized.bin",
        "backend": "NPU_QNN_HTP",
        "performance_mode": "DEFAULT",
        "log_level": "INFO",
        "debug_mode": false })");
    CHECK(c.name == "scrfd_quantized");
    CHECK(c.backends == std::vector<Backend>({ Backend::Npu }));

    const ModelConfig a = parse_model_config(R"({"modelPath": "/m/detect.tflite"})");
    CHECK(a.name == "detect");
    CHECK(a.backends.size() == 4); // AUTO when absent

    const ModelConfig l = parse_model_config(R"({"modelPath": "/m/x.tflite", "backend": ["XNNPACK","CPU"], "threads": 2})");
    CHECK(l.backends == std::vector<Backend>({ Backend::Xnnpack, Backend::Cpu }));
    CHECK(threads_for(l, 4) == 2);

    CHECK_THROWS_CODE(parse_model_config(R"({"name":"other","modelPath":"/m/x.tflite"})"), ErrorCode::InvalidConfig);
    CHECK_THROWS_CODE(parse_model_config(R"({"modelPath":"/m/x.tflite","backend":"TPU"})"), ErrorCode::InvalidConfig);
    CHECK_THROWS_CODE(parse_model_config(R"({"modelPath":"/m/x.tflite","threads":-1})"), ErrorCode::InvalidConfig);
    CHECK_THROWS_CODE(parse_model_config("not json"), ErrorCode::InvalidConfig);
    CHECK_THROWS_CODE(parse_model_config(R"({"name":"x"})"), ErrorCode::InvalidConfig);
}

TEST("errors: every code survives the D-Bus name round trip")
{
    for (int i = 1; i <= static_cast<int>(ErrorCode::Internal); ++i) {
        const auto code = static_cast<ErrorCode>(i);
        CHECK(error_code_from_string(dbus_error_name(code)) == code);
    }
    CHECK_THROWS_CODE(raise(ErrorCode::OutOfMemory, "x"), ErrorCode::OutOfMemory);
    bool memory = false;
    try {
        raise(ErrorCode::OutOfMemory, "x");
    } catch (const MemoryException &) {
        memory = true;
    } catch (...) {
    }
    CHECK(memory);
}

int main() { return check::run_all(); }
