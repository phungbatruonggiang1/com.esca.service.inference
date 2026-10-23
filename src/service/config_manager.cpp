#include "config_manager.hpp"

#include <fstream>

#include "logger.hpp"

namespace esca::inference {

namespace {

std::string trim(const std::string &s)
{
    const size_t a = s.find_first_not_of(" \t\r");
    const size_t b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

} // namespace

ConfigManager &ConfigManager::instance()
{
    static ConfigManager manager;
    return manager;
}

bool ConfigManager::load_file(const std::string &path, std::string &error)
{
    std::ifstream in(path);
    // A missing file means "defaults", which is the normal case.
    if (!in)
        return true;

    ServiceConfig &c = config_;
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        ++number;
        line = trim(line.substr(0, line.find('#')));
        if (line.empty())
            continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            error = path + ":" + std::to_string(number) + ": expected key = value";
            return false;
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        try {
            if (key == "config_dir") c.config_dir = value;
            else if (key == "workers") c.workers = static_cast<unsigned>(std::stoul(value));
            else if (key == "pool_threads") c.pool_threads = static_cast<unsigned>(std::stoul(value));
            else if (key == "cpu_slots") c.cpu_slots = std::stoi(value);
            else if (key == "memory_reserve_mb") c.memory_reserve_mb = std::stoul(value);
            else if (key == "pressure_threshold") c.pressure_threshold = std::stod(value);
            else if (key == "eviction") c.eviction = value;
            else if (key == "scheduling") c.scheduling = value;
            else if (key == "shm_group") c.shm_group = value;
            else if (key == "log_level") c.log_level = value;
            else ESCA_LOG_WARN("%s:%d: unknown key \"%s\"", path.c_str(), number, key.c_str());
        } catch (const std::exception &) {
            error = path + ":" + std::to_string(number) + ": bad value for " + key;
            return false;
        }
    }
    return true;
}

} // namespace esca::inference
