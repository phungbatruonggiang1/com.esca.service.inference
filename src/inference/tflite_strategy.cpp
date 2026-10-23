#include "tflite_strategy.hpp"

#include <sys/stat.h>

#include <cstring>

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"

#include "esca/inference/errors.hpp"

namespace esca::inference {

namespace {

DataType from_tflite(TfLiteType type)
{
    switch (type) {
    case kTfLiteFloat32: return DataType::Float32;
    case kTfLiteFloat16: return DataType::Float16;
    case kTfLiteInt8: return DataType::Int8;
    case kTfLiteUInt8: return DataType::UInt8;
    case kTfLiteInt16: return DataType::Int16;
    case kTfLiteInt32: return DataType::Int32;
    case kTfLiteInt64: return DataType::Int64;
    case kTfLiteBool: return DataType::Bool;
    default: return DataType::Unknown;
    }
}

TensorInfo describe(const TfLiteTensor *t)
{
    TensorInfo info;
    info.name = t->name ? t->name : "";
    info.type = from_tflite(t->type);
    // dims_signature keeps the -1 of a dynamic dimension; dims holds the
    // shape currently allocated, which is what a buffer must match.
    for (int i = 0; t->dims && i < t->dims->size; ++i)
        info.shape.push_back(t->dims->data[i]);
    info.scale = t->params.scale;
    info.zero_point = t->params.zero_point;
    info.bytes = t->bytes;
    return info;
}

size_t file_size(const std::string &path)
{
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? static_cast<size_t>(st.st_size) : 0;
}

} // namespace

struct TfliteStrategy::Impl {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    TfLiteDelegate *delegate = nullptr;

    void reset()
    {
        // The interpreter refers to the delegate, so it goes first.
        interpreter.reset();
        if (delegate) {
            TfLiteXNNPackDelegateDelete(delegate);
            delegate = nullptr;
        }
        model.reset();
    }

    ~Impl() { reset(); }
};

TfliteStrategy::TfliteStrategy(std::string model_path, Backend backend, int threads)
    : impl_(std::make_unique<Impl>()), path_(std::move(model_path)), backend_(backend),
      threads_(threads < 1 ? 1 : threads)
{
}

TfliteStrategy::~TfliteStrategy() = default;

bool TfliteStrategy::supports(Backend backend)
{
    // The GPU delegate needs OpenCL or a GLES 3.1 compute stack this image
    // does not build TFLite against, and a Pi has no NPU.
    return backend == Backend::Cpu || backend == Backend::Xnnpack;
}

void TfliteStrategy::load()
{
    if (loaded())
        return;
    if (!supports(backend_))
        raise(ErrorCode::BackendUnavailable,
              std::string("TensorFlow Lite has no ") + to_string(backend_) + " backend in this build");

    impl_->model = tflite::FlatBufferModel::BuildFromFile(path_.c_str());
    if (!impl_->model)
        raise(ErrorCode::ModelLoadFailed, "cannot read model " + path_);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    if (tflite::InterpreterBuilder(*impl_->model, resolver)(&impl_->interpreter) != kTfLiteOk
        || !impl_->interpreter) {
        impl_->reset();
        raise(ErrorCode::ModelLoadFailed, "cannot build an interpreter for " + path_);
    }
    impl_->interpreter->SetNumThreads(threads_);

    if (backend_ == Backend::Xnnpack) {
        TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
        options.num_threads = threads_;
        impl_->delegate = TfLiteXNNPackDelegateCreate(&options);
        // Not every model is fully supported by XNNPACK. Unlike esca-visiond,
        // which quietly fell back to the reference kernels, this says so: the
        // allocator then picks the CPU backend explicitly and reports it.
        if (!impl_->delegate
            || impl_->interpreter->ModifyGraphWithDelegate(impl_->delegate) != kTfLiteOk) {
            impl_->reset();
            raise(ErrorCode::BackendUnavailable, "XNNPACK cannot run " + path_);
        }
    }

    if (impl_->interpreter->AllocateTensors() != kTfLiteOk) {
        impl_->reset();
        raise(ErrorCode::OutOfMemory, "cannot allocate tensors for " + path_);
    }
}

void TfliteStrategy::unload() { impl_->reset(); }

bool TfliteStrategy::loaded() const { return impl_->interpreter != nullptr; }

std::vector<TensorInfo> TfliteStrategy::inputs() const
{
    std::vector<TensorInfo> out;
    if (!loaded())
        return out;
    for (int index : impl_->interpreter->inputs())
        out.push_back(describe(impl_->interpreter->tensor(index)));
    return out;
}

std::vector<TensorInfo> TfliteStrategy::outputs() const
{
    std::vector<TensorInfo> out;
    if (!loaded())
        return out;
    for (int index : impl_->interpreter->outputs())
        out.push_back(describe(impl_->interpreter->tensor(index)));
    return out;
}

void TfliteStrategy::resize_inputs(const std::vector<std::vector<int32_t>> &shapes)
{
    if (!loaded())
        raise(ErrorCode::Internal, "resize before load");
    const auto &indices = impl_->interpreter->inputs();
    bool changed = false;
    for (size_t i = 0; i < shapes.size() && i < indices.size(); ++i) {
        if (shapes[i].empty())
            continue;
        const TfLiteTensor *t = impl_->interpreter->tensor(indices[i]);
        std::vector<int> current(t->dims->data, t->dims->data + t->dims->size);
        std::vector<int> wanted(shapes[i].begin(), shapes[i].end());
        if (current == wanted)
            continue;
        if (impl_->interpreter->ResizeInputTensor(indices[i], wanted) != kTfLiteOk)
            raise(ErrorCode::InvalidArgument, "input " + std::to_string(i) + " cannot take that shape");
        changed = true;
    }
    if (changed && impl_->interpreter->AllocateTensors() != kTfLiteOk)
        raise(ErrorCode::OutOfMemory, "cannot allocate tensors after a resize");
}

void TfliteStrategy::run(const std::vector<ConstBuffer> &inputs,
                         const std::vector<MutableBuffer> &outputs)
{
    if (!loaded())
        raise(ErrorCode::Internal, "run before load");
    tflite::Interpreter &interp = *impl_->interpreter;

    const auto &in_index = interp.inputs();
    if (inputs.size() != in_index.size())
        raise(ErrorCode::InvalidArgument, "model takes " + std::to_string(in_index.size())
                                              + " inputs, got " + std::to_string(inputs.size()));
    for (size_t i = 0; i < inputs.size(); ++i) {
        TfLiteTensor *t = interp.tensor(in_index[i]);
        if (inputs[i].size < t->bytes)
            raise(ErrorCode::BufferError, "input " + std::to_string(i) + " buffer holds "
                                              + std::to_string(inputs[i].size) + " bytes, tensor needs "
                                              + std::to_string(t->bytes));
        std::memcpy(t->data.raw, inputs[i].data, t->bytes);
    }

    if (interp.Invoke() != kTfLiteOk)
        raise(ErrorCode::InferenceFailed, "Invoke failed for " + path_);

    const auto &out_index = interp.outputs();
    for (size_t i = 0; i < outputs.size() && i < out_index.size(); ++i) {
        const TfLiteTensor *t = interp.tensor(out_index[i]);
        if (outputs[i].size < t->bytes)
            raise(ErrorCode::BufferError, "output " + std::to_string(i) + " buffer holds "
                                              + std::to_string(outputs[i].size) + " bytes, tensor has "
                                              + std::to_string(t->bytes));
        std::memcpy(outputs[i].data, t->data.raw, t->bytes);
    }
}

size_t TfliteStrategy::memory_footprint() const
{
    // The mapped model plus every tensor the interpreter allocated: a fair
    // upper bound for the budget, which only needs to be roughly right.
    size_t bytes = file_size(path_);
    if (loaded())
        for (size_t i = 0; i < impl_->interpreter->tensors_size(); ++i)
            bytes += impl_->interpreter->tensor(static_cast<int>(i))->bytes;
    return bytes;
}

// --- builder -------------------------------------------------------------------

bool InferenceStrategyBuilder::available(Backend backend) { return TfliteStrategy::supports(backend); }

std::unique_ptr<InferenceStrategy> InferenceStrategyBuilder::build(const ModelConfig &config,
                                                                    Backend backend, int threads)
{
    const std::string &path = config.model_path;
    const size_t dot = path.find_last_of('.');
    const std::string ext = dot == std::string::npos ? "" : path.substr(dot);
    if (ext == ".tflite")
        return std::make_unique<TfliteStrategy>(path, backend, threads);
    // .onnx would be ONNX Runtime and .bin QNN; neither has a recipe in this
    // image's layers.
    raise(ErrorCode::BackendUnavailable, "no engine for \"" + ext + "\" models in this build ("
                                             + config.name + ")");
}

} // namespace esca::inference
