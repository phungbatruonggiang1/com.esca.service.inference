// One error taxonomy from the engine to the application.
//
// Each code has a D-Bus error name, so a failure raised deep in the engine
// reaches a client as vn.sparc.esca.Inference1.Error.<Name> and the client
// library turns it back into the same exception type. A caller can tell
// "no such model" from "out of memory" without parsing a message.

#pragma once

#include <stdexcept>
#include <string>

namespace esca::inference {

enum class ErrorCode {
    Ok = 0,
    // model
    ModelNotFound,
    ModelAlreadyRegistered,
    ModelLoadFailed,
    ModelBusy,
    // configuration and files
    InvalidConfig,
    FileNotFound,
    // memory
    OutOfMemory,
    BufferError,
    // contexts and inference
    ContextNotFound,
    InvalidArgument,
    InferenceFailed,
    BackendUnavailable,
    // transport
    ServiceUnavailable,
    Timeout,
    Internal,
};

/// "ModelNotFound", the last component of the D-Bus error name.
const char *to_string(ErrorCode code);
ErrorCode error_code_from_string(const std::string &name);

/// model, config, file, memory, inference, service: for grouping in logs.
const char *category(ErrorCode code);

/// vn.sparc.esca.Inference1.Error.<code>
std::string dbus_error_name(ErrorCode code);

class InferenceException : public std::runtime_error {
public:
    InferenceException(ErrorCode code, const std::string &message)
        : std::runtime_error(message), code_(code)
    {
    }
    ErrorCode code() const { return code_; }

private:
    ErrorCode code_;
};

// The specialisations exist so a handler can catch one family without
// switching on codes; each still carries its precise code.

class ModelException : public InferenceException {
public:
    using InferenceException::InferenceException;
};

class ConfigException : public InferenceException {
public:
    explicit ConfigException(const std::string &message)
        : InferenceException(ErrorCode::InvalidConfig, message)
    {
    }
};

class FileException : public InferenceException {
public:
    explicit FileException(const std::string &message)
        : InferenceException(ErrorCode::FileNotFound, message)
    {
    }
};

class MemoryException : public InferenceException {
public:
    using InferenceException::InferenceException;
};

class RunException : public InferenceException {
public:
    using InferenceException::InferenceException;
};

/// Throws the specialisation matching the code.
[[noreturn]] void raise(ErrorCode code, const std::string &message);

} // namespace esca::inference
