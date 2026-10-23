#include "logger.hpp"

#include <mutex>

namespace esca::inference::log {

Level &threshold()
{
    static Level level = Level::Info;
    return level;
}

bool set_threshold(const std::string &name)
{
    if (name == "DEBUG") threshold() = Level::Debug;
    else if (name == "INFO") threshold() = Level::Info;
    else if (name == "WARN" || name == "WARNING") threshold() = Level::Warn;
    else if (name == "ERROR") threshold() = Level::Error;
    else return false;
    return true;
}

void write(Level level, const char *fmt, ...)
{
    if (level < threshold())
        return;
    static std::mutex mutex;
    static const char *const tags[] = { "debug", "info", "warn", "error" };

    char line[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(line, sizeof(line), fmt, args);
    va_end(args);

    // One line per message, written under a lock, so the worker threads'
    // messages do not interleave in the journal.
    std::lock_guard<std::mutex> lock(mutex);
    std::fprintf(stderr, "esca-inference [%s] %s\n", tags[static_cast<int>(level)], line);
}

} // namespace esca::inference::log
