#include "model_registration.hpp"

#include <dirent.h>
#include <unistd.h>

#include <algorithm>

#include "esca/inference/errors.hpp"
#include "logger.hpp"
#include "model_config.hpp"

namespace esca::inference {

void ModelRegistrationHelper::check_model_file(const ModelConfig &config)
{
    if (::access(config.model_path.c_str(), R_OK) != 0)
        throw FileException(config.name + ": model file " + config.model_path
                            + " is missing or unreadable");
}

RegistrationReport ModelRegistrationHelper::scan(const std::string &dir)
{
    RegistrationReport report;
    DIR *d = ::opendir(dir.c_str());
    if (!d) {
        report.errors.push_back(dir + ": cannot open the directory");
        return report;
    }
    std::vector<std::string> files;
    while (const dirent *e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() > 5 && name.compare(name.size() - 5, 5, ".json") == 0)
            files.push_back(dir + "/" + name);
    }
    ::closedir(d);
    std::sort(files.begin(), files.end());

    for (const std::string &file : files) {
        try {
            const ModelConfig config = load_model_config(file);
            if (models_.contains(config.name)) {
                // The same file seen again on Refresh() is not an error;
                // another file claiming the same name is.
                if (models_.config(config.name).source == file)
                    report.unchanged.push_back(config.name);
                else
                    report.errors.push_back(file + ": \"" + config.name
                                            + "\" is already registered from another file");
                continue;
            }
            check_model_file(config);
            models_.register_model(config);
            report.registered.push_back(config.name);
        } catch (const std::exception &e) {
            report.errors.push_back(file + ": " + e.what());
        }
    }
    for (const std::string &error : report.errors)
        ESCA_LOG_WARN("skipped %s", error.c_str());
    return report;
}

std::string ModelRegistrationHelper::register_json(const std::string &json)
{
    const ModelConfig config = parse_model_config(json, "");
    check_model_file(config);
    models_.register_model(config);
    return config.name;
}

} // namespace esca::inference
