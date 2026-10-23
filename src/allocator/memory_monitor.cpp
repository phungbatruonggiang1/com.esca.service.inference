#include "memory_monitor.hpp"

#include <cstdio>
#include <cstring>

namespace esca::inference {

size_t MemoryMonitor::available_bytes() const
{
    FILE *f = std::fopen("/proc/meminfo", "r");
    if (!f)
        return 0;
    char line[256];
    size_t kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            std::sscanf(line + 13, "%zu", &kb);
            break;
        }
    }
    std::fclose(f);
    return kb * 1024;
}

std::optional<double> MemoryMonitor::pressure() const
{
    FILE *f = std::fopen("/proc/pressure/memory", "r");
    if (!f)
        return std::nullopt;
    char line[256];
    std::optional<double> some;
    // some avg10=0.00 avg60=0.00 avg300=0.00 total=0
    if (std::fgets(line, sizeof(line), f)) {
        double avg10 = 0;
        if (std::sscanf(line, "some avg10=%lf", &avg10) == 1)
            some = avg10;
    }
    std::fclose(f);
    return some;
}

} // namespace esca::inference
