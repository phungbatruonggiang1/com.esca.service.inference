#include "model_manager.hpp"

#include "esca/inference/errors.hpp"
#include "logger.hpp"

namespace esca::inference {

namespace {
std::string names(const std::vector<Backend> &backends)
{
    std::string out;
    for (Backend b : backends)
        out += (out.empty() ? "" : ",") + std::string(to_string(b));
    return out;
}

constexpr size_t kMiB = 1u << 20;
} // namespace

ModelManager::ModelManager(StrategyFactory factory, BackendProbe available,
                           std::unique_ptr<EvictionPolicy> eviction,
                           std::shared_ptr<MemoryMonitor> memory, ModelManagerOptions options)
    : factory_(std::move(factory)), memory_(std::move(memory)), options_(options),
      selector_(std::move(available), options.cpu_slots),
      eviction_(eviction ? std::move(eviction) : std::make_unique<NoEvictionPolicy>())
{
    watcher_ = std::thread([this] { watch_loop(); });
    preloader_ = std::thread([this] { preload_loop(); });
}

ModelManager::~ModelManager()
{
    {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    watcher_.join();
    preloader_.join();
}

void ModelManager::set_listener(ModelStateListener listener)
{
    std::lock_guard<std::mutex> lock(mutex_);
    listener_ = std::move(listener);
}

void ModelManager::notify(const std::string &name, ModelState state, const std::string &backend)
{
    ModelStateListener listener;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        listener = listener_;
    }
    // Outside the lock: the listener emits a D-Bus signal, and must be free
    // to call back into list().
    if (listener)
        listener(name, state, backend);
}

std::shared_ptr<Model> ModelManager::find(const std::string &name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = models_.find(name);
    if (it == models_.end())
        raise(ErrorCode::ModelNotFound, "no model named \"" + name + "\"");
    return it->second;
}

bool ModelManager::contains(const std::string &name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return models_.count(name) != 0;
}

void ModelManager::register_model(const ModelConfig &config)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (models_.count(config.name))
            raise(ErrorCode::ModelAlreadyRegistered, "\"" + config.name + "\" is already registered");
        models_[config.name] =
            std::make_shared<Model>(config, factory_, threads_for(config, options_.cores));
    }
    ESCA_LOG_INFO("registered %s (%s, backends %s)", config.name.c_str(),
                  config.model_path.c_str(), names(config.backends).c_str());
    notify(config.name, ModelState::Registered, "");
    if (config.preload)
        preload_async(config.name);
}

void ModelManager::unregister_model(const std::string &name)
{
    std::shared_ptr<Model> model = find(name);
    if (model->in_flight() > 0)
        raise(ErrorCode::ModelBusy, name + " has requests in flight");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        models_.erase(name);
    }
    model->unload();
    ESCA_LOG_INFO("unregistered %s", name.c_str());
}

std::vector<ModelSummary> ModelManager::list() const
{
    std::vector<std::shared_ptr<Model>> all;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[name, model] : models_)
            all.push_back(model);
    }
    std::vector<ModelSummary> out;
    for (const auto &m : all) {
        ModelSummary s;
        s.name = m->name();
        s.state = m->state();
        s.preferred = m->config().backends;
        s.loaded = m->loaded_backends();
        s.model_path = m->config().model_path;
        s.last_error = m->last_error();
        s.bytes = m->footprint();
        s.in_flight = m->in_flight();
        out.push_back(std::move(s));
    }
    return out;
}

ModelConfig ModelManager::config(const std::string &name) const { return find(name)->config(); }

std::vector<LoadedModel> ModelManager::loaded_models(const std::string &except) const
{
    std::vector<std::shared_ptr<Model>> all;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[name, model] : models_)
            if (name != except)
                all.push_back(model);
    }
    std::vector<LoadedModel> out;
    for (const auto &m : all)
        if (!m->loaded_backends().empty())
            out.push_back(LoadedModel { m->name(), m->footprint(), m->last_used(), m->in_flight() });
    return out;
}

void ModelManager::ensure_memory(const Model &model)
{
    const size_t need = model.estimate_load_bytes();
    auto shortfall = [&]() -> size_t {
        const size_t available = memory_->available_bytes();
        const size_t wanted = need + options_.memory_reserve_bytes;
        return available >= wanted ? 0 : wanted - available;
    };

    size_t missing = shortfall();
    if (missing == 0)
        return;

    const std::vector<LoadedModel> loaded = loaded_models(model.name());
    std::vector<std::string> victims;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        victims = eviction_->select_victims(loaded, missing);
    }
    for (const std::string &victim : victims) {
        std::shared_ptr<Model> m;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = models_.find(victim);
            if (it != models_.end())
                m = it->second;
        }
        // Checked again here: a request may have arrived since the policy
        // looked, and a busy model is never unloaded under it.
        if (!m || m->in_flight() > 0)
            continue;
        ESCA_LOG_WARN("memory short by %zu MiB for %s: unloading %s", missing / kMiB,
                      model.name().c_str(), victim.c_str());
        m->unload();
        notify(victim, ModelState::Registered, "evicted");
        missing = shortfall();
        if (missing == 0)
            return;
    }
    raise(ErrorCode::OutOfMemory,
          "not enough memory to load " + model.name() + ": " + std::to_string(need / kMiB)
              + " MiB needed, " + std::to_string(memory_->available_bytes() / kMiB)
              + " MiB available with " + std::to_string(options_.memory_reserve_bytes / kMiB)
              + " MiB kept in reserve");
}

Backend ModelManager::load_locked_out(const std::shared_ptr<Model> &model)
{
    // Already loaded somewhere usable: that is enough for load().
    for (Backend b : selector_.candidates(*model))
        if (model->loaded(b))
            return b;

    std::string failures;
    for (Backend b : selector_.candidates(*model)) {
        try {
            {
                std::lock_guard<std::mutex> memory(memory_mutex_);
                ensure_memory(*model);
                model->load(b);
            }
            ESCA_LOG_INFO("loaded %s on %s (%zu KiB)", model->name().c_str(), to_string(b),
                          model->footprint() / 1024);
            notify(model->name(), ModelState::Loaded, to_string(b));
            return b;
        } catch (const InferenceException &e) {
            // Memory will not improve by trying another backend; anything
            // else might.
            if (e.code() == ErrorCode::OutOfMemory) {
                notify(model->name(), ModelState::Failed, to_string(b));
                throw;
            }
            ESCA_LOG_WARN("%s on %s: %s", model->name().c_str(), to_string(b), e.what());
            failures += std::string(failures.empty() ? "" : "; ") + to_string(b) + ": " + e.what();
        }
    }
    notify(model->name(), ModelState::Failed, "");
    raise(ErrorCode::ModelLoadFailed, model->name() + " could not be loaded on any backend"
                                          + (failures.empty() ? std::string() : " (" + failures + ")"));
}

Backend ModelManager::load(const std::string &name) { return load_locked_out(find(name)); }

void ModelManager::preload_async(const std::string &name)
{
    {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        preload_queue_.push_back(name);
    }
    wake_.notify_all();
}

void ModelManager::preload_loop()
{
    for (;;) {
        std::string name;
        {
            std::unique_lock<std::mutex> lock(thread_mutex_);
            wake_.wait(lock, [this] { return stopping_ || !preload_queue_.empty(); });
            if (stopping_)
                return;
            name = preload_queue_.front();
            preload_queue_.pop_front();
        }
        try {
            load(name);
        } catch (const std::exception &e) {
            ESCA_LOG_WARN("preload %s: %s", name.c_str(), e.what());
        }
    }
}

void ModelManager::unload(const std::string &name)
{
    std::shared_ptr<Model> model = find(name);
    if (model->in_flight() > 0)
        raise(ErrorCode::ModelBusy, name + " has requests in flight");
    model->unload();
    ESCA_LOG_INFO("unloaded %s", name.c_str());
    notify(name, ModelState::Registered, "");
}

ModelMetadata ModelManager::metadata(const std::string &name)
{
    std::shared_ptr<Model> model = find(name);
    if (!model->has_metadata())
        load_locked_out(model);
    return model->metadata();
}

ModelMetadata ModelManager::describe(const std::string &name,
                                     const std::vector<std::vector<int32_t>> &shapes)
{
    std::shared_ptr<Model> model = find(name);
    const Backend b = load_locked_out(model);
    return model->describe(b, shapes);
}

void ModelManager::acquire(const std::string &name) { find(name)->acquire(); }

void ModelManager::release(const std::string &name)
{
    std::shared_ptr<Model> model;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = models_.find(name);
        if (it != models_.end())
            model = it->second;
    }
    if (model)
        model->release();
}

RunOutcome ModelManager::run(const std::string &name, const std::vector<std::vector<int32_t>> &shapes,
                             const std::vector<ConstBuffer> &inputs,
                             const std::vector<MutableBuffer> &outputs, Backend &used)
{
    std::shared_ptr<Model> model = find(name);

    // Pick per request; then make sure that backend is loaded, falling
    // through the candidates when it cannot be.
    std::vector<Backend> order;
    const Backend first = selector_.select(*model);
    order.push_back(first);
    for (Backend b : selector_.candidates(*model))
        if (b != first)
            order.push_back(b);

    std::string failures;
    for (Backend b : order) {
        if (!model->loaded(b)) {
            try {
                std::lock_guard<std::mutex> memory(memory_mutex_);
                ensure_memory(*model);
                model->load(b);
            } catch (const InferenceException &e) {
                if (e.code() == ErrorCode::OutOfMemory)
                    throw;
                failures += std::string(failures.empty() ? "" : "; ") + to_string(b) + ": " + e.what();
                continue;
            }
            notify(name, ModelState::Loaded, to_string(b));
        }

        selector_.acquire(b);
        try {
            RunOutcome outcome = model->run(b, shapes, inputs, outputs);
            selector_.release(b);
            used = b;
            return outcome;
        } catch (...) {
            selector_.release(b);
            throw;
        }
    }
    raise(ErrorCode::BackendUnavailable,
          name + ": no backend could run the request" + (failures.empty() ? "" : " (" + failures + ")"));
}

void ModelManager::set_eviction_policy(std::unique_ptr<EvictionPolicy> policy)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (policy)
        eviction_ = std::move(policy);
}

std::string ModelManager::eviction_policy() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return eviction_->name();
}

size_t ModelManager::loaded_bytes() const
{
    size_t bytes = 0;
    for (const LoadedModel &m : loaded_models(""))
        bytes += m.bytes;
    return bytes;
}

void ModelManager::relieve_pressure()
{
    const size_t available = memory_->available_bytes();
    const std::optional<double> pressure = memory_->pressure();
    const bool short_of_memory = available < options_.memory_reserve_bytes;
    const bool stalled = pressure && *pressure >= options_.pressure_threshold;
    if (!short_of_memory && !stalled)
        return;

    // One model a pass: unloading is not free to undo, and the next pass two
    // seconds later sees whether this one was enough.
    const std::vector<LoadedModel> loaded = loaded_models("");
    std::vector<std::string> victims;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        victims = eviction_->select_victims(loaded, 1);
    }
    if (victims.empty())
        return;

    std::lock_guard<std::mutex> memory(memory_mutex_);
    std::shared_ptr<Model> m;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = models_.find(victims.front());
        if (it != models_.end())
            m = it->second;
    }
    if (!m || m->in_flight() > 0)
        return;
    ESCA_LOG_WARN("memory pressure (%zu MiB available%s): unloading idle %s", available / kMiB,
                  stalled ? ", tasks stalling" : "", m->name().c_str());
    m->unload();
    notify(m->name(), ModelState::Registered, "evicted");
}

void ModelManager::watch_loop()
{
    std::unique_lock<std::mutex> lock(thread_mutex_);
    while (!stopping_) {
        wake_.wait_for(lock, options_.watch_period);
        if (stopping_)
            return;
        lock.unlock();
        relieve_pressure();
        lock.lock();
    }
}

} // namespace esca::inference
