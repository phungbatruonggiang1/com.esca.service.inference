#include "model_config.hpp"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "esca/inference/errors.hpp"

namespace esca::inference {

using nlohmann::json;

namespace {

[[noreturn]] void bad(const std::string &source, const std::string &what)
{
    throw ConfigException((source.empty() ? std::string("model config") : source) + ": " + what);
}

std::string string_field(const json &doc, const char *key, const std::string &source,
                         bool required, const std::string &fallback = {})
{
    const auto it = doc.find(key);
    if (it == doc.end() || it->is_null()) {
        if (required)
            bad(source, std::string("missing \"") + key + "\"");
        return fallback;
    }
    if (!it->is_string())
        bad(source, std::string("\"") + key + "\" must be a string");
    return it->get<std::string>();
}

/// "debug_mode" may be written as a string or a boolean; take either.
bool bool_field(const json &doc, const char *key, const std::string &source, bool fallback)
{
    const auto it = doc.find(key);
    if (it == doc.end() || it->is_null())
        return fallback;
    if (it->is_boolean())
        return it->get<bool>();
    if (it->is_string()) {
        const std::string v = it->get<std::string>();
        if (v == "true" || v == "1") return true;
        if (v == "false" || v == "0") return false;
    }
    bad(source, std::string("\"") + key + "\" must be true or false");
}

std::vector<Backend> backends_field(const json &doc, const std::string &source)
{
    const auto it = doc.find("backend");
    if (it == doc.end() || it->is_null())
        return auto_backends();

    std::vector<std::string> names;
    if (it->is_string())
        names.push_back(it->get<std::string>());
    else if (it->is_array())
        for (const json &v : *it) {
            if (!v.is_string())
                bad(source, "\"backend\" entries must be strings");
            names.push_back(v.get<std::string>());
        }
    else
        bad(source, "\"backend\" must be a string or a list of strings");

    std::vector<Backend> out;
    for (const std::string &name : names) {
        if (name == "AUTO" || name == "auto") {
            for (Backend b : auto_backends())
                out.push_back(b);
            continue;
        }
        const auto b = backend_from_string(name);
        if (!b)
            bad(source, "unknown backend \"" + name + "\"");
        out.push_back(*b);
    }
    if (out.empty())
        bad(source, "\"backend\" is empty");
    return out;
}

} // namespace

const std::vector<Backend> &auto_backends()
{
    static const std::vector<Backend> order { Backend::Npu, Backend::Gpu, Backend::Xnnpack,
                                              Backend::Cpu };
    return order;
}

std::string model_name_from_path(const std::string &path)
{
    const size_t slash = path.find_last_of('/');
    std::string file = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = file.find_last_of('.');
    return dot == std::string::npos || dot == 0 ? file : file.substr(0, dot);
}

ModelConfig parse_model_config(const std::string &text, const std::string &source)
{
    json doc;
    try {
        doc = json::parse(text);
    } catch (const json::parse_error &e) {
        bad(source, std::string("not JSON: ") + e.what());
    }
    if (!doc.is_object())
        bad(source, "expected a JSON object");

    ModelConfig config;
    config.source = source;
    config.model_path = string_field(doc, "modelPath", source, true);
    config.name = string_field(doc, "name", source, false, model_name_from_path(config.model_path));
    if (config.name.empty())
        bad(source, "\"name\" is empty");

    // The name is the file's own name. Two configs
    // pointing at one file under different names would load it twice.
    if (config.name != model_name_from_path(config.model_path))
        bad(source, "\"name\" \"" + config.name + "\" must match the model file name \""
                        + model_name_from_path(config.model_path) + "\"");

    config.backends = backends_field(doc, source);

    const std::string mode = string_field(doc, "performance_mode", source, false, "DEFAULT");
    const auto parsed = performance_mode_from_string(mode);
    if (!parsed)
        bad(source, "unknown performance_mode \"" + mode + "\"");
    config.performance_mode = *parsed;

    if (const auto it = doc.find("threads"); it != doc.end() && !it->is_null()) {
        if (!it->is_number_integer() || it->get<int>() < 0 || it->get<int>() > 64)
            bad(source, "\"threads\" must be an integer from 0 to 64");
        config.threads = it->get<int>();
    }

    config.preload = bool_field(doc, "preload", source, false);
    config.debug_mode = bool_field(doc, "debug_mode", source, false);
    config.log_level = string_field(doc, "log_level", source, false, "INFO");
    return config;
}

ModelConfig load_model_config(const std::string &path)
{
    std::ifstream in(path);
    if (!in)
        throw FileException("cannot read " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return parse_model_config(text.str(), path);
}

std::string to_json(const ModelConfig &config)
{
    json backends = json::array();
    for (Backend b : config.backends)
        backends.push_back(to_string(b));
    json doc = {
        { "name", config.name },
        { "modelPath", config.model_path },
        { "backend", backends },
        { "performance_mode", to_string(config.performance_mode) },
        { "threads", config.threads },
        { "preload", config.preload },
        { "log_level", config.log_level },
        { "debug_mode", config.debug_mode },
    };
    if (!config.source.empty())
        doc["source"] = config.source;
    return doc.dump();
}

} // namespace esca::inference
