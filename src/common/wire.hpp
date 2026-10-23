// The D-Bus interface, as both ends see it: names, and how a tensor
// description travels. Internal to this project -- applications use the
// client library -- but shared by its service and client so the two cannot
// drift apart.

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <sdbus-c++/sdbus-c++.h>

#include "esca/inference/errors.hpp"
#include "esca/inference/types.hpp"

namespace esca::inference::wire {

constexpr const char *kService = "vn.sparc.esca.Inference1";
constexpr const char *kPath = "/vn/sparc/esca/Inference1";
constexpr const char *kInterface = "vn.sparc.esca.Inference1";

/// (name, dtype, shape, scale, zero_point, bytes)
using Tensor = sdbus::Struct<std::string, std::string, std::vector<int32_t>, double, int64_t, uint64_t>;

/// (name, state, preferred backends, loaded backends, model path, bytes, last error)
using ModelRow =
    sdbus::Struct<std::string, std::string, std::string, std::string, std::string, uint64_t, std::string>;

inline Tensor to_wire(const TensorInfo &t)
{
    return Tensor { t.name, std::string(to_string(t.type)), t.shape, t.scale, t.zero_point,
                    static_cast<uint64_t>(t.bytes) };
}

inline TensorInfo from_wire(const Tensor &w)
{
    TensorInfo t;
    t.name = w.get<0>();
    t.type = data_type_from_string(w.get<1>());
    t.shape = w.get<2>();
    t.scale = w.get<3>();
    t.zero_point = w.get<4>();
    t.bytes = static_cast<size_t>(w.get<5>());
    return t;
}

inline std::vector<Tensor> to_wire(const std::vector<TensorInfo> &ts)
{
    std::vector<Tensor> out;
    for (const TensorInfo &t : ts)
        out.push_back(to_wire(t));
    return out;
}

inline std::vector<TensorInfo> from_wire(const std::vector<Tensor> &ws)
{
    std::vector<TensorInfo> out;
    for (const Tensor &w : ws)
        out.push_back(from_wire(w));
    return out;
}

inline sdbus::Error to_dbus(ErrorCode code, const std::string &message)
{
    return sdbus::Error(sdbus::Error::Name { dbus_error_name(code) }, message);
}

/// The client side: a D-Bus error from this service back into the exception
/// the service raised; any other D-Bus error is ServiceUnavailable.
[[noreturn]] inline void rethrow(const sdbus::Error &e)
{
    const std::string name = e.getName();
    const std::string prefix = "vn.sparc.esca.Inference1.Error.";
    if (name.rfind(prefix, 0) == 0)
        raise(error_code_from_string(name), e.getMessage());
    if (name == "org.freedesktop.DBus.Error.NoReply" || name == "org.freedesktop.DBus.Error.Timeout")
        raise(ErrorCode::Timeout, e.getMessage());
    raise(ErrorCode::ServiceUnavailable, name + ": " + e.getMessage());
}

} // namespace esca::inference::wire
