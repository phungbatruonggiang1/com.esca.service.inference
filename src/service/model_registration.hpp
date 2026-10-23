// Turning configuration files into registered models.
//
// The model registration helper: scans the configuration directory,
// checks each file and the model it names, registers what is valid, and
// reports every file it skipped with the reason -- a model that silently
// fails to appear is the hardest kind of misconfiguration to find.

#pragma once

#include <string>
#include <vector>

#include "model_manager.hpp"

namespace esca::inference {

struct RegistrationReport {
    std::vector<std::string> registered;
    std::vector<std::string> unchanged; ///< already registered from the same file
    std::vector<std::string> errors;    ///< "file: reason"
};

class ModelRegistrationHelper {
public:
    explicit ModelRegistrationHelper(ModelManager &models) : models_(models) {}

    /// Every *.json in `dir`, in name order.
    RegistrationReport scan(const std::string &dir);

    /// One configuration from the bus. Returns the model name; throws
    /// ConfigException, FileException or ModelAlreadyRegistered.
    std::string register_json(const std::string &json);

    /// Throws FileException unless the model file exists and is readable.
    static void check_model_file(const ModelConfig &config);

private:
    ModelManager &models_;
};

} // namespace esca::inference
