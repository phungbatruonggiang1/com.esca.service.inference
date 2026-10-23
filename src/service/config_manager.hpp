// Service defaults, from /etc/esca/inference.conf and the command line.
//
// The configuration manager: one place that holds the bus and object names,
// the pool sizes, the model configuration directory and the log level.

#pragma once

#include <string>

namespace esca::inference {

struct ServiceConfig {
    std::string service_name = "vn.sparc.esca.Inference1";
    std::string object_path = "/vn/sparc/esca/Inference1";
    bool session_bus = false;

    /// Where model configurations (*.json) are read from at startup and on
    /// Refresh().
    std::string config_dir = "/etc/esca/inference/models.d";

    unsigned workers = 2;     ///< inference worker threads
    unsigned pool_threads = 2; ///< async context creation, loads
    int cpu_slots = 2;         ///< CPU-backend requests at once before it is busy

    size_t memory_reserve_mb = 128;
    double pressure_threshold = 10.0;
    std::string eviction = "lru";
    std::string scheduling = "fifo";

    /// Group the tensor buffers belong to; clients outside it cannot map them.
    std::string shm_group = "inference";
    std::string log_level = "INFO";
};

class ConfigManager {
public:
    static ConfigManager &instance();

    /// Reads `key = value` lines; unknown keys are reported, not fatal.
    /// Returns false only if the file exists and cannot be read.
    bool load_file(const std::string &path, std::string &error);

    ServiceConfig &config() { return config_; }

private:
    ConfigManager() = default;
    ServiceConfig config_;
};

} // namespace esca::inference
