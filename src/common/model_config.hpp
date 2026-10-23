// A model's JSON configuration, parsed and checked.
//
// Used by the service, which reads a directory of them, and by the client's
// local mode, which reads one. Parsing is defensive: a malformed file names
// the field that is wrong instead of being half-accepted.

#pragma once

#include <string>

#include "esca/inference/types.hpp"

namespace esca::inference {

/// Throws ConfigException naming the offending field. `source` is only used
/// in messages and recorded in the result.
ModelConfig parse_model_config(const std::string &json, const std::string &source = {});

/// Reads and parses a file. Throws FileException if it cannot be read.
ModelConfig load_model_config(const std::string &path);

/// The configuration back as JSON, for GetModelConfig and logs.
std::string to_json(const ModelConfig &config);

/// "detect" for "/usr/share/esca/models/detect.tflite".
std::string model_name_from_path(const std::string &path);

/// What "AUTO" means: every backend, fastest first. The device selector
/// drops those this machine does not have.
const std::vector<Backend> &auto_backends();

} // namespace esca::inference
