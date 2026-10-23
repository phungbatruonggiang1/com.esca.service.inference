// The two ways a Session runs a model, and the builder that picks one.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <sstream>
#include <thread>

#include "esca/inference/client.hpp"
#include "model_config.hpp"
#include "strategy.hpp"

namespace esca::inference {

namespace {

/// One of the service's shared-memory buffers, mapped here.
class MappedBuffer {
public:
    explicit MappedBuffer(const std::string &name) : name_(name)
    {
        const int fd = ::shm_open(name.c_str(), O_RDWR, 0);
        if (fd < 0)
            raise(ErrorCode::BufferError, "cannot open " + name + ": " + std::strerror(errno)
                                              + (errno == EACCES ? " (is this user in the inference group?)" : ""));
        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            raise(ErrorCode::BufferError, "cannot stat " + name);
        }
        size_ = static_cast<size_t>(st.st_size);
        data_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (data_ == MAP_FAILED)
            raise(ErrorCode::BufferError, "cannot map " + name + ": " + std::strerror(errno));
    }
    ~MappedBuffer()
    {
        if (data_ != MAP_FAILED)
            ::munmap(data_, size_);
    }
    MappedBuffer(const MappedBuffer &) = delete;
    MappedBuffer &operator=(const MappedBuffer &) = delete;

    const std::string &name() const { return name_; }
    void *data() const { return data_; }
    size_t size() const { return size_; }

private:
    std::string name_;
    void *data_ = MAP_FAILED;
    size_t size_ = 0;
};

/// The service-based strategy: owns a context in
/// esca-inferenced and maps its buffers, so the application writes inputs
/// straight into the memory the service reads.
class ServiceBasedSession : public Session {
public:
    ServiceBasedSession(std::shared_ptr<InferenceServiceClient> client, const std::string &model)
        : client_(std::move(client)), model_(model)
    {
        adopt(client_->create_context(model));
    }

    ~ServiceBasedSession() override
    {
        // Best effort: a service that is gone has dropped the context anyway.
        try {
            client_->destroy_context(context_.id);
        } catch (const std::exception &) {
        }
    }

    const std::string &model() const override { return model_; }
    std::vector<TensorBuffer> &inputs() override { return inputs_; }
    std::vector<TensorBuffer> &outputs() override { return outputs_; }
    bool remote() const override { return true; }

    InferenceTiming run(int priority) override
    {
        std::vector<TensorInfo> shapes;
        const InferenceTiming timing = client_->infer(context_.id, priority, &shapes);
        for (size_t i = 0; i < shapes.size() && i < outputs_.size(); ++i)
            outputs_[i].info = shapes[i];
        return timing;
    }

    void run_async(InferenceCallback done, int priority) override
    {
        client_->infer_async(context_.id, priority, std::move(done));
    }

    void set_input_shape(size_t index, const std::vector<int32_t> &shape) override
    {
        adopt(client_->set_input_shape(context_.id, static_cast<uint32_t>(index), shape));
    }

private:
    /// Maps the context's buffers, keeping mappings whose name did not change.
    void adopt(InferenceServiceClient::Context context)
    {
        auto map = [this](const std::vector<std::string> &names) {
            std::vector<std::shared_ptr<MappedBuffer>> out;
            for (const std::string &n : names) {
                std::shared_ptr<MappedBuffer> found;
                for (const auto &m : maps_)
                    if (m->name() == n)
                        found = m;
                out.push_back(found ? found : std::make_shared<MappedBuffer>(n));
            }
            return out;
        };
        const auto in = map(context.input_buffers);
        const auto out = map(context.output_buffers);

        inputs_.clear();
        outputs_.clear();
        for (size_t i = 0; i < in.size(); ++i)
            inputs_.push_back(TensorBuffer { context.inputs[i], in[i]->data(), in[i]->size() });
        for (size_t i = 0; i < out.size(); ++i)
            outputs_.push_back(TensorBuffer { context.outputs[i], out[i]->data(), out[i]->size() });

        maps_ = in;
        maps_.insert(maps_.end(), out.begin(), out.end());
        context_ = std::move(context);
    }

    std::shared_ptr<InferenceServiceClient> client_;
    std::string model_;
    InferenceServiceClient::Context context_;
    std::vector<std::shared_ptr<MappedBuffer>> maps_;
    std::vector<TensorBuffer> inputs_;
    std::vector<TensorBuffer> outputs_;
};

/// The model in this process, through libesca-inference directly: no
/// service, no shared memory. Tries the configured backends in order.
class LocalSession : public Session {
public:
    explicit LocalSession(const ModelConfig &config) : config_(config)
    {
        const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
        std::string failures;
        for (Backend b : config.backends) {
            if (!InferenceStrategyBuilder::available(b))
                continue;
            try {
                auto s = InferenceStrategyBuilder::build(config, b, threads_for(config, cores));
                s->load();
                strategy_ = std::move(s);
                backend_ = b;
                break;
            } catch (const InferenceException &e) {
                if (e.code() != ErrorCode::BackendUnavailable)
                    throw;
                failures += std::string(failures.empty() ? "" : "; ") + e.what();
            }
        }
        if (!strategy_)
            raise(ErrorCode::BackendUnavailable,
                  config.name + ": no backend could load it locally" + (failures.empty() ? "" : " (" + failures + ")"));
        for (const TensorInfo &t : strategy_->inputs())
            shapes_.push_back(t.shape);
        reallocate();
    }

    ~LocalSession() override
    {
        if (pending_.valid())
            pending_.wait();
    }

    const std::string &model() const override { return config_.name; }
    std::vector<TensorBuffer> &inputs() override { return inputs_; }
    std::vector<TensorBuffer> &outputs() override { return outputs_; }
    bool remote() const override { return false; }

    InferenceTiming run(int) override
    {
        std::vector<ConstBuffer> in;
        std::vector<MutableBuffer> out;
        for (const TensorBuffer &b : inputs_)
            in.push_back({ b.data, b.size });
        for (const TensorBuffer &b : outputs_)
            out.push_back({ b.data, b.size });
        const auto started = std::chrono::steady_clock::now();
        strategy_->resize_inputs(shapes_);
        strategy_->run(in, out);
        const std::vector<TensorInfo> produced = strategy_->outputs();
        for (size_t i = 0; i < produced.size() && i < outputs_.size(); ++i)
            outputs_[i].info = produced[i];
        return InferenceTiming {
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(), 0,
            backend_ };
    }

    void run_async(InferenceCallback done, int priority) override
    {
        if (pending_.valid())
            pending_.wait();
        pending_ = std::async(std::launch::async, [this, done = std::move(done), priority] {
            try {
                const InferenceTiming t = run(priority);
                done(ErrorCode::Ok, "", t);
            } catch (const InferenceException &e) {
                done(e.code(), e.what(), InferenceTiming {});
            }
        });
    }

    void set_input_shape(size_t index, const std::vector<int32_t> &shape) override
    {
        if (index >= shapes_.size())
            raise(ErrorCode::InvalidArgument, "no input " + std::to_string(index));
        shapes_[index] = shape;
        strategy_->resize_inputs(shapes_);
        reallocate();
    }

private:
    void reallocate()
    {
        const auto in = strategy_->inputs();
        const auto out = strategy_->outputs();
        in_memory_.assign(in.size(), {});
        out_memory_.assign(out.size(), {});
        inputs_.clear();
        outputs_.clear();
        for (size_t i = 0; i < in.size(); ++i) {
            in_memory_[i].resize(in[i].bytes);
            inputs_.push_back({ in[i], in_memory_[i].data(), in_memory_[i].size() });
        }
        for (size_t i = 0; i < out.size(); ++i) {
            out_memory_[i].resize(out[i].bytes);
            outputs_.push_back({ out[i], out_memory_[i].data(), out_memory_[i].size() });
        }
    }

    ModelConfig config_;
    std::unique_ptr<InferenceStrategy> strategy_;
    Backend backend_ = Backend::Cpu;
    std::vector<std::vector<int32_t>> shapes_;
    std::vector<std::vector<uint8_t>> in_memory_, out_memory_;
    std::vector<TensorBuffer> inputs_, outputs_;
    std::future<void> pending_;
};

std::string read_file(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw FileException("cannot read " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

} // namespace

InferenceBuilder &InferenceBuilder::model(std::string name)
{
    model_ = std::move(name);
    return *this;
}

InferenceBuilder &InferenceBuilder::config(std::string path)
{
    config_ = std::move(path);
    return *this;
}

InferenceBuilder &InferenceBuilder::mode(Mode mode)
{
    mode_ = mode;
    return *this;
}

InferenceBuilder &InferenceBuilder::session_bus(bool session)
{
    session_bus_ = session;
    return *this;
}

InferenceBuilder &InferenceBuilder::config_dir(std::string dir)
{
    config_dir_ = std::move(dir);
    return *this;
}

std::unique_ptr<Session> InferenceBuilder::build()
{
    std::string name = model_;
    if (name.empty() && !config_.empty())
        name = load_model_config(config_).name;
    if (name.empty())
        raise(ErrorCode::InvalidArgument, "InferenceBuilder needs model() or config()");

    Mode mode = mode_;
    if (mode == Mode::Auto)
        mode = InferenceServiceClient::available(session_bus_) ? Mode::Service : Mode::Local;

    if (mode == Mode::Service) {
        auto client = std::make_shared<InferenceServiceClient>(session_bus_);
        if (!config_.empty()) {
            bool known = false;
            for (const ModelSummaryInfo &m : client->list_models())
                known = known || m.name == name;
            if (!known)
                client->register_model(read_file(config_));
        }
        return std::make_unique<ServiceBasedSession>(std::move(client), name);
    }

    const std::string path = config_.empty() ? config_dir_ + "/" + name + ".json" : config_;
    return std::make_unique<LocalSession>(load_model_config(path));
}

} // namespace esca::inference
