#include "esca/inference/types.hpp"

#include <algorithm>
#include <cctype>

#include "esca/inference/errors.hpp"

namespace esca::inference {

namespace {

std::string upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

} // namespace

const char *to_string(DataType type)
{
    switch (type) {
    case DataType::Float32: return "float32";
    case DataType::Float16: return "float16";
    case DataType::Int8: return "int8";
    case DataType::UInt8: return "uint8";
    case DataType::Int16: return "int16";
    case DataType::Int32: return "int32";
    case DataType::Int64: return "int64";
    case DataType::Bool: return "bool";
    case DataType::Unknown: break;
    }
    return "unknown";
}

DataType data_type_from_string(const std::string &name)
{
    for (DataType t : { DataType::Float32, DataType::Float16, DataType::Int8, DataType::UInt8,
                        DataType::Int16, DataType::Int32, DataType::Int64, DataType::Bool })
        if (name == to_string(t))
            return t;
    return DataType::Unknown;
}

size_t element_size(DataType type)
{
    switch (type) {
    case DataType::Float32:
    case DataType::Int32: return 4;
    case DataType::Float16:
    case DataType::Int16: return 2;
    case DataType::Int8:
    case DataType::UInt8:
    case DataType::Bool: return 1;
    case DataType::Int64: return 8;
    case DataType::Unknown: break;
    }
    return 0;
}

size_t element_count(const std::vector<int32_t> &shape)
{
    size_t count = 1;
    for (int32_t d : shape)
        count *= d > 0 ? static_cast<size_t>(d) : 1;
    return count;
}

const char *to_string(Backend backend)
{
    switch (backend) {
    case Backend::Cpu: return "CPU";
    case Backend::Xnnpack: return "XNNPACK";
    case Backend::Gpu: return "GPU";
    case Backend::Npu: return "NPU";
    }
    return "CPU";
}

std::optional<Backend> backend_from_string(const std::string &name)
{
    const std::string n = upper(name);
    if (n == "CPU") return Backend::Cpu;
    if (n == "XNNPACK" || n == "CPU_XNNPACK") return Backend::Xnnpack;
    if (n == "GPU" || n == "GPU_OPENCL" || n == "GPU_GLES") return Backend::Gpu;
    // Vendor names such as NPU_QNN_HTP: anything NPU means the NPU.
    if (n.rfind("NPU", 0) == 0) return Backend::Npu;
    return std::nullopt;
}

const char *to_string(PerformanceMode mode)
{
    switch (mode) {
    case PerformanceMode::Default: return "DEFAULT";
    case PerformanceMode::HighPerformance: return "HIGH_PERFORMANCE";
    case PerformanceMode::PowerSaving: return "POWER_SAVING";
    }
    return "DEFAULT";
}

std::optional<PerformanceMode> performance_mode_from_string(const std::string &name)
{
    const std::string n = upper(name);
    if (n == "DEFAULT") return PerformanceMode::Default;
    if (n == "HIGH_PERFORMANCE" || n == "HIGH") return PerformanceMode::HighPerformance;
    if (n == "POWER_SAVING" || n == "LOW") return PerformanceMode::PowerSaving;
    return std::nullopt;
}

int threads_for(const ModelConfig &config, unsigned cores)
{
    if (config.threads > 0)
        return config.threads;
    const int all = cores == 0 ? 1 : static_cast<int>(cores);
    switch (config.performance_mode) {
    // Every core: the caller has said this model matters most.
    case PerformanceMode::HighPerformance: return all;
    case PerformanceMode::PowerSaving: return 1;
    // One core left for whatever draws the result, as esca-visiond did.
    case PerformanceMode::Default: break;
    }
    return std::max(1, all - 1);
}

// --- errors ------------------------------------------------------------------

const char *to_string(ErrorCode code)
{
    switch (code) {
    case ErrorCode::Ok: return "Ok";
    case ErrorCode::ModelNotFound: return "ModelNotFound";
    case ErrorCode::ModelAlreadyRegistered: return "ModelAlreadyRegistered";
    case ErrorCode::ModelLoadFailed: return "ModelLoadFailed";
    case ErrorCode::ModelBusy: return "ModelBusy";
    case ErrorCode::InvalidConfig: return "InvalidConfig";
    case ErrorCode::FileNotFound: return "FileNotFound";
    case ErrorCode::OutOfMemory: return "OutOfMemory";
    case ErrorCode::BufferError: return "BufferError";
    case ErrorCode::ContextNotFound: return "ContextNotFound";
    case ErrorCode::InvalidArgument: return "InvalidArgument";
    case ErrorCode::InferenceFailed: return "InferenceFailed";
    case ErrorCode::BackendUnavailable: return "BackendUnavailable";
    case ErrorCode::ServiceUnavailable: return "ServiceUnavailable";
    case ErrorCode::Timeout: return "Timeout";
    case ErrorCode::Internal: return "Internal";
    }
    return "Internal";
}

ErrorCode error_code_from_string(const std::string &name)
{
    // Accept the full D-Bus name as well as the bare code.
    const std::string bare = name.substr(name.rfind('.') == std::string::npos ? 0 : name.rfind('.') + 1);
    for (int i = 0; i <= static_cast<int>(ErrorCode::Internal); ++i) {
        const auto code = static_cast<ErrorCode>(i);
        if (bare == to_string(code))
            return code;
    }
    return ErrorCode::Internal;
}

const char *category(ErrorCode code)
{
    switch (code) {
    case ErrorCode::ModelNotFound:
    case ErrorCode::ModelAlreadyRegistered:
    case ErrorCode::ModelLoadFailed:
    case ErrorCode::ModelBusy: return "model";
    case ErrorCode::InvalidConfig: return "config";
    case ErrorCode::FileNotFound: return "file";
    case ErrorCode::OutOfMemory:
    case ErrorCode::BufferError: return "memory";
    case ErrorCode::ContextNotFound:
    case ErrorCode::InvalidArgument:
    case ErrorCode::InferenceFailed:
    case ErrorCode::BackendUnavailable: return "inference";
    case ErrorCode::ServiceUnavailable:
    case ErrorCode::Timeout: return "service";
    case ErrorCode::Ok:
    case ErrorCode::Internal: break;
    }
    return "internal";
}

std::string dbus_error_name(ErrorCode code)
{
    return std::string("vn.sparc.esca.Inference1.Error.") + to_string(code);
}

void raise(ErrorCode code, const std::string &message)
{
    switch (code) {
    case ErrorCode::ModelNotFound:
    case ErrorCode::ModelAlreadyRegistered:
    case ErrorCode::ModelLoadFailed:
    case ErrorCode::ModelBusy: throw ModelException(code, message);
    case ErrorCode::InvalidConfig: throw ConfigException(message);
    case ErrorCode::FileNotFound: throw FileException(message);
    case ErrorCode::OutOfMemory:
    case ErrorCode::BufferError: throw MemoryException(code, message);
    case ErrorCode::ContextNotFound:
    case ErrorCode::InvalidArgument:
    case ErrorCode::InferenceFailed:
    case ErrorCode::BackendUnavailable: throw RunException(code, message);
    case ErrorCode::Ok:
    case ErrorCode::ServiceUnavailable:
    case ErrorCode::Timeout:
    case ErrorCode::Internal: break;
    }
    throw InferenceException(code, message);
}

} // namespace esca::inference
