// What every layer of the inference stack agrees on: tensors, backends and
// the model configuration a JSON file describes.
//
// Shared by the engine (libesca-inference), the allocator, the service and
// the client library, so a tensor described by the service reads the same in
// the application that maps it.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace esca::inference {

enum class DataType { Float32, Float16, Int8, UInt8, Int16, Int32, Int64, Bool, Unknown };

const char *to_string(DataType type);
DataType data_type_from_string(const std::string &name);
size_t element_size(DataType type);

/// One input or output of a model, as a client needs it to fill or read the
/// shared-memory buffer behind it.
struct TensorInfo {
    std::string name;
    DataType type = DataType::Unknown;
    std::vector<int32_t> shape;
    /// Affine quantisation: real = scale * (q - zero_point). Zero scale means
    /// the tensor is not quantised.
    double scale = 0.0;
    int64_t zero_point = 0;
    /// Bytes for the current shape. A dynamic dimension (-1) counts as 1
    /// until a context sets a concrete shape.
    size_t bytes = 0;
};

/// Number of elements in a shape; dynamic dimensions count as 1.
size_t element_count(const std::vector<int32_t> &shape);

/// Where a model runs. On a Raspberry Pi 4 only the CPU ones exist; GPU and
/// NPU are understood so that a configuration written for other hardware is
/// accepted and falls back rather than failing to parse.
enum class Backend { Cpu, Xnnpack, Gpu, Npu };

const char *to_string(Backend backend);
std::optional<Backend> backend_from_string(const std::string &name);

enum class PerformanceMode { Default, HighPerformance, PowerSaving };

const char *to_string(PerformanceMode mode);
std::optional<PerformanceMode> performance_mode_from_string(const std::string &name);

/// One model, as its JSON configuration declares it.
///
///     {
///       "name": "detect",
///       "modelPath": "/usr/share/esca/models/detect.tflite",
///       "backend": "AUTO",                 or "XNNPACK", or ["XNNPACK", "CPU"]
///       "performance_mode": "DEFAULT",     HIGH_PERFORMANCE, POWER_SAVING
///       "threads": 3,                      optional, 0 = from the mode
///       "preload": false,                  load at registration
///       "log_level": "INFO",
///       "debug_mode": false
///     }
///
/// The name must match the model file's name without directory or
/// extension: /usr/share/esca/models/detect.tflite is "detect".
struct ModelConfig {
    std::string name;
    std::string model_path;
    /// In order of preference. "AUTO" expands to every backend this build
    /// knows, fastest first.
    std::vector<Backend> backends;
    PerformanceMode performance_mode = PerformanceMode::Default;
    int threads = 0;
    bool preload = false;
    std::string log_level = "INFO";
    bool debug_mode = false;
    /// Where the configuration came from, for messages; empty when it was
    /// registered over the bus.
    std::string source;
};

/// Threads to give an interpreter for a configuration on this machine.
int threads_for(const ModelConfig &config, unsigned cores);

} // namespace esca::inference
