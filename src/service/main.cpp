// esca-inferenced: runs models for the rest of the board.

#include <csignal>
#include <cstdio>
#include <string>

#include <sdbus-c++/sdbus-c++.h>

#include "config_manager.hpp"
#include "logger.hpp"
#include "service.hpp"

namespace {

sdbus::IConnection *g_connection = nullptr;

void handle_signal(int)
{
    if (g_connection)
        g_connection->leaveEventLoop();
}

int usage()
{
    std::fprintf(stderr,
                 "usage: esca-inferenced [options]\n"
                 "  --config FILE       service settings (default /etc/esca/inference.conf)\n"
                 "  --config-dir DIR    model configurations (*.json)\n"
                 "  --workers N         inference threads\n"
                 "  --eviction POLICY   lru or none\n"
                 "  --scheduling POLICY fifo or priority\n"
                 "  --log-level LEVEL   DEBUG, INFO, WARN, ERROR\n"
                 "  --session           use the session bus (development)\n");
    return 2;
}

} // namespace

int main(int argc, char **argv)
{
    using namespace esca::inference;

    std::string config_file = "/etc/esca/inference.conf";
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--config")
            config_file = argv[i + 1];

    ConfigManager &manager = ConfigManager::instance();
    std::string error;
    if (!manager.load_file(config_file, error)) {
        std::fprintf(stderr, "esca-inferenced: %s\n", error.c_str());
        return 1;
    }
    ServiceConfig &config = manager.config();

    // The command line wins over the file.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        try {
            if (arg == "--config") next();
            else if (arg == "--config-dir") config.config_dir = next();
            else if (arg == "--workers") config.workers = static_cast<unsigned>(std::stoul(next()));
            else if (arg == "--eviction") config.eviction = next();
            else if (arg == "--scheduling") config.scheduling = next();
            else if (arg == "--log-level") config.log_level = next();
            else if (arg == "--session") config.session_bus = true;
            else return usage();
        } catch (const std::exception &) {
            return usage();
        }
    }
    if (!log::set_threshold(config.log_level))
        ESCA_LOG_WARN("unknown log level \"%s\"", config.log_level.c_str());
    if (!make_eviction_policy(config.eviction) || !make_scheduling_policy(config.scheduling)) {
        std::fprintf(stderr, "esca-inferenced: unknown eviction or scheduling policy\n");
        return 2;
    }

    try {
        const sdbus::ServiceName name { config.service_name };
        auto connection = config.session_bus ? sdbus::createSessionBusConnection(name)
                                             : sdbus::createSystemBusConnection(name);
        g_connection = connection.get();
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        InferenceService service(*connection, config);
        service.start();

        ESCA_LOG_INFO("ready on %s (%u workers, %s scheduling, %s eviction, %zu MiB reserve)",
                      config.service_name.c_str(), config.workers, config.scheduling.c_str(),
                      config.eviction.c_str(), config.memory_reserve_mb);
        connection->enterEventLoop();
        ESCA_LOG_INFO("shutting down");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "esca-inferenced: fatal: %s\n", e.what());
        return 1;
    }
}
