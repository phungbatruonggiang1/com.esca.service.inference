// How much memory is left, as the kernel sees it.

#pragma once

#include <cstddef>
#include <optional>

namespace esca::inference {

class MemoryMonitor {
public:
    virtual ~MemoryMonitor() = default;

    /// MemAvailable: what can be allocated without swapping, page cache
    /// included. MemFree would say a busy board is always nearly full.
    virtual size_t available_bytes() const;

    /// PSI "some avg10" for memory, in percent: how much of the last ten
    /// seconds some task spent stalled on memory. Absent on kernels without
    /// CONFIG_PSI.
    virtual std::optional<double> pressure() const;
};

} // namespace esca::inference
