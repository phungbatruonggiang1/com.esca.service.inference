#include "model.hpp"

#include <sys/stat.h>

#include "esca/inference/errors.hpp"

namespace esca::inference {

const char *to_string(ModelState state)
{
    switch (state) {
    case ModelState::Registered: return "registered";
    case ModelState::Loading: return "loading";
    case ModelState::Loaded: return "loaded";
    case ModelState::Failed: return "failed";
    }
    return "registered";
}

Model::Model(ModelConfig config, StrategyFactory factory, int threads)
    : config_(std::move(config)), factory_(std::move(factory)), threads_(threads)
{
}

ModelState Model::state() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

std::string Model::last_error() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
}

void Model::load(Backend backend)
{
    std::lock_guard<std::mutex> loading(load_mutex_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (executors_.count(backend))
            return;
        if (unusable_.count(backend))
            raise(ErrorCode::BackendUnavailable,
                  name() + " cannot run on " + to_string(backend));
        state_ = ModelState::Loading;
    }

    // Built outside the lock: loading takes long, and list() or metadata()
    // from another thread must not wait on it.
    auto executor = std::make_shared<Executor>();
    try {
        executor->strategy = factory_(config_, backend, threads_);
        executor->strategy->load();
    } catch (const InferenceException &e) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (e.code() == ErrorCode::BackendUnavailable)
            unusable_.insert(backend);
        last_error_ = e.what();
        state_ = executors_.empty() ? ModelState::Failed : ModelState::Loaded;
        throw;
    } catch (const std::exception &e) {
        // A framework throwing its own type is still a failed load, and must
        // reach the caller as one.
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = e.what();
        state_ = executors_.empty() ? ModelState::Failed : ModelState::Loaded;
        raise(ErrorCode::ModelLoadFailed, name() + ": " + e.what());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_metadata_) {
        metadata_.inputs = executor->strategy->inputs();
        metadata_.outputs = executor->strategy->outputs();
        has_metadata_ = true;
    }
    executors_[backend] = std::move(executor);
    state_ = ModelState::Loaded;
    last_error_.clear();
}

bool Model::loaded(Backend backend) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return executors_.count(backend) != 0;
}

std::vector<Backend> Model::loaded_backends() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Backend> out;
    for (const auto &[backend, executor] : executors_)
        out.push_back(backend);
    return out;
}

bool Model::unusable(Backend backend) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return unusable_.count(backend) != 0;
}

void Model::unload()
{
    std::map<Backend, std::shared_ptr<Executor>> dropped;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dropped.swap(executors_);
        state_ = ModelState::Registered;
    }
    // Each runtime is released under its own run lock, so one still
    // finishing an inference completes before it goes.
    for (auto &[backend, executor] : dropped) {
        std::lock_guard<std::mutex> run(executor->run_mutex);
        executor->strategy->unload();
    }
}

bool Model::has_metadata() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return has_metadata_;
}

ModelMetadata Model::metadata() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_metadata_)
        raise(ErrorCode::ModelLoadFailed, name() + " has not been loaded yet");
    return metadata_;
}

RunOutcome Model::run(Backend backend, const std::vector<std::vector<int32_t>> &shapes,
                      const std::vector<ConstBuffer> &inputs,
                      const std::vector<MutableBuffer> &outputs)
{
    std::shared_ptr<Executor> executor;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = executors_.find(backend);
        if (it == executors_.end())
            raise(ErrorCode::Internal, name() + " is not loaded on " + to_string(backend));
        executor = it->second;
        last_used_ = std::chrono::steady_clock::now();
    }

    std::lock_guard<std::mutex> run(executor->run_mutex);
    if (!executor->strategy->loaded())
        raise(ErrorCode::ModelLoadFailed, name() + " was unloaded while waiting to run");
    executor->strategy->resize_inputs(shapes);
    executor->strategy->run(inputs, outputs);
    return RunOutcome { executor->strategy->outputs() };
}

ModelMetadata Model::describe(Backend backend, const std::vector<std::vector<int32_t>> &shapes)
{
    std::shared_ptr<Executor> executor;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = executors_.find(backend);
        if (it == executors_.end())
            raise(ErrorCode::Internal, name() + " is not loaded on " + to_string(backend));
        executor = it->second;
    }
    // Under the run lock: resizing the interpreter must not race a run.
    // Every run resizes to its own context's shapes first, so leaving the
    // interpreter at these shapes is harmless.
    std::lock_guard<std::mutex> run(executor->run_mutex);
    executor->strategy->resize_inputs(shapes);
    return ModelMetadata { executor->strategy->inputs(), executor->strategy->outputs() };
}

size_t Model::footprint() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t bytes = 0;
    for (const auto &[backend, executor] : executors_)
        bytes += executor->strategy->memory_footprint();
    return bytes;
}

size_t Model::estimate_load_bytes() const
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &[backend, executor] : executors_)
            return executor->strategy->memory_footprint();
    }
    // Never loaded: the file plus as much again for tensors and the arena.
    struct stat st {};
    if (::stat(config_.model_path.c_str(), &st) == 0)
        return static_cast<size_t>(st.st_size) * 2;
    return 0;
}

std::chrono::steady_clock::time_point Model::last_used() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return last_used_;
}

void Model::touch()
{
    std::lock_guard<std::mutex> lock(mutex_);
    last_used_ = std::chrono::steady_clock::now();
}

} // namespace esca::inference
