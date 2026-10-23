#include "context_manager.hpp"

#include <unistd.h>

#include "esca/inference/errors.hpp"
#include "logger.hpp"

namespace esca::inference {

ContextManager::ContextManager(ModelManager &models, InferenceScheduler &scheduler,
                               BufferAllocator &buffers)
    : models_(models), scheduler_(scheduler), buffers_(buffers)
{
}

std::shared_ptr<ContextManager::Context> ContextManager::find(const std::string &id,
                                                              const std::string &owner) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = contexts_.find(id);
    if (it == contexts_.end())
        raise(ErrorCode::ContextNotFound, "no context \"" + id + "\"");
    if (!owner.empty() && it->second->owner != owner)
        raise(ErrorCode::InvalidArgument, "context \"" + id + "\" belongs to another client");
    return it->second;
}

void ContextManager::allocate(Context &c, const ModelMetadata &metadata)
{
    // Names carry the pid, the context and a generation, so a buffer
    // replaced by a resize never collides with the one a client may still
    // have mapped, and leftovers from a crashed service are recognisable.
    const std::string prefix = "/esca.inf." + std::to_string(::getpid()) + "." + c.id + "."
                               + std::to_string(c.generation) + ".";
    std::vector<std::shared_ptr<SharedBuffer>> in, out;
    for (size_t i = 0; i < metadata.inputs.size(); ++i) {
        // Keep a buffer that is already the right size: the client's mapping
        // of it stays valid.
        if (i < c.input_buffers.size() && c.input_buffers[i]->size() >= metadata.inputs[i].bytes
            && c.input_buffers[i]->size() <= metadata.inputs[i].bytes + 4096)
            in.push_back(c.input_buffers[i]);
        else
            in.push_back(buffers_.allocate(prefix + "i" + std::to_string(i), metadata.inputs[i].bytes));
    }
    for (size_t i = 0; i < metadata.outputs.size(); ++i) {
        if (i < c.output_buffers.size() && c.output_buffers[i]->size() >= metadata.outputs[i].bytes
            && c.output_buffers[i]->size() <= metadata.outputs[i].bytes + 4096)
            out.push_back(c.output_buffers[i]);
        else
            out.push_back(buffers_.allocate(prefix + "o" + std::to_string(i), metadata.outputs[i].bytes));
    }
    c.inputs = metadata.inputs;
    c.outputs = metadata.outputs;
    c.input_buffers = std::move(in);
    c.output_buffers = std::move(out);
}

ContextView ContextManager::view_of(const Context &c)
{
    ContextView v { c.id, c.model, c.inputs, c.outputs, {}, {} };
    for (const auto &b : c.input_buffers)
        v.input_buffers.push_back(b->name());
    for (const auto &b : c.output_buffers)
        v.output_buffers.push_back(b->name());
    return v;
}

ContextView ContextManager::create(const std::string &model, const std::string &owner)
{
    const ModelMetadata metadata = models_.metadata(model);

    auto c = std::make_shared<Context>();
    c->id = "ctx" + std::to_string(next_id_++);
    c->model = model;
    c->owner = owner;
    for (const TensorInfo &t : metadata.inputs)
        c->shapes.push_back(t.shape);
    allocate(*c, metadata);

    ContextView view = view_of(*c);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        contexts_[c->id] = c;
    }
    ESCA_LOG_INFO("context %s on %s for %s", c->id.c_str(), model.c_str(),
                  owner.empty() ? "the service" : owner.c_str());
    return view;
}

void ContextManager::destroy(const std::string &id, const std::string &owner)
{
    find(id, owner); // existence and ownership
    std::lock_guard<std::mutex> lock(mutex_);
    // Buffers an inference is still writing into stay mapped until it ends:
    // the request holds its own references to them.
    contexts_.erase(id);
}

size_t ContextManager::destroy_owned_by(const std::string &owner)
{
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = 0;
    for (auto it = contexts_.begin(); it != contexts_.end();) {
        if (it->second->owner == owner) {
            it = contexts_.erase(it);
            ++n;
        } else {
            ++it;
        }
    }
    return n;
}

ContextView ContextManager::view(const std::string &id, const std::string &owner) const
{
    std::shared_ptr<Context> c = find(id, owner);
    std::lock_guard<std::mutex> lock(c->mutex);
    return view_of(*c);
}

ContextView ContextManager::set_input_shape(const std::string &id, const std::string &owner,
                                            size_t index, const std::vector<int32_t> &shape)
{
    std::shared_ptr<Context> c = find(id, owner);
    if (c->busy)
        raise(ErrorCode::ModelBusy, "context \"" + id + "\" has an inference in flight");

    std::lock_guard<std::mutex> lock(c->mutex);
    if (index >= c->shapes.size())
        raise(ErrorCode::InvalidArgument, "the model has " + std::to_string(c->shapes.size())
                                              + " inputs; there is no input " + std::to_string(index));
    if (shape.size() != c->shapes[index].size())
        raise(ErrorCode::InvalidArgument, "input " + std::to_string(index) + " has rank "
                                              + std::to_string(c->shapes[index].size()));
    for (int32_t d : shape)
        if (d <= 0)
            raise(ErrorCode::InvalidArgument, "dimensions must be positive");

    std::vector<std::vector<int32_t>> shapes = c->shapes;
    shapes[index] = shape;
    const ModelMetadata metadata = models_.describe(c->model, shapes);
    ++c->generation;
    c->shapes = std::move(shapes);
    allocate(*c, metadata);
    return view_of(*c);
}

uint64_t ContextManager::infer(const std::string &id, const std::string &owner, int priority,
                               std::function<void(const InferenceResult &)> done)
{
    std::shared_ptr<Context> c = find(id, owner);
    bool expected = false;
    if (!c->busy.compare_exchange_strong(expected, true))
        raise(ErrorCode::ModelBusy, "context \"" + id + "\" has an inference in flight");

    InferenceRequest request;
    {
        std::lock_guard<std::mutex> lock(c->mutex);
        request.model = c->model;
        request.priority = priority;
        request.shapes = c->shapes;
        for (const auto &b : c->input_buffers)
            request.inputs.push_back(ConstBuffer { b->data(), b->size() });
        for (const auto &b : c->output_buffers)
            request.outputs.push_back(MutableBuffer { b->data(), b->size() });
    }
    // The request keeps the context, and through it the buffers, alive
    // until it is answered, however the client behaves meanwhile.
    request.done = [c, done = std::move(done)](const InferenceResult &result) {
        if (result.code == ErrorCode::Ok) {
            std::lock_guard<std::mutex> lock(c->mutex);
            c->outputs = result.outputs;
        }
        c->busy = false;
        done(result);
    };
    try {
        return scheduler_.submit(std::move(request));
    } catch (...) {
        c->busy = false;
        throw;
    }
}

size_t ContextManager::count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return contexts_.size();
}

} // namespace esca::inference
