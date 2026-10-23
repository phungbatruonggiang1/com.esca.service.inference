// esca-inferctl: the inference service from a terminal.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "esca/inference/client.hpp"

using namespace esca::inference;

namespace {

int usage()
{
    std::fprintf(stderr,
                 "usage: esca-inferctl [--session] COMMAND\n"
                 "  list                         registered models and their state\n"
                 "  info MODEL                   input and output tensors (loads the model)\n"
                 "  config MODEL                 the model's configuration\n"
                 "  register FILE.json           register a model\n"
                 "  unregister MODEL\n"
                 "  load MODEL | unload MODEL\n"
                 "  refresh                      rescan the configuration directory\n"
                 "  stats                        scheduler and memory figures\n"
                 "  policy scheduling|eviction NAME\n"
                 "  run MODEL [--input FILE] [--repeat N] [--async] [--local] [--config FILE]\n"
                 "            [--priority P]     run it and print timing and outputs\n");
    return 2;
}

std::string shape_of(const std::vector<int32_t> &shape)
{
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i)
        s += (i ? "," : "") + std::to_string(shape[i]);
    return s + "]";
}

void print_tensors(const char *title, const std::vector<TensorInfo> &ts)
{
    std::printf("%s:\n", title);
    for (size_t i = 0; i < ts.size(); ++i) {
        const TensorInfo &t = ts[i];
        std::printf("  %zu %-28s %-8s %-16s %8zu bytes", i, t.name.c_str(), to_string(t.type),
                    shape_of(t.shape).c_str(), t.bytes);
        if (t.scale != 0)
            std::printf("  q(scale %.6g, zero %lld)", t.scale, static_cast<long long>(t.zero_point));
        std::printf("\n");
    }
}

/// The first few values of an output, dequantised when quantised.
void print_values(const TensorBuffer &b)
{
    const size_t n = std::min<size_t>(8, element_count(b.info.shape));
    std::printf("  %-28s", b.info.name.c_str());
    for (size_t i = 0; i < n; ++i) {
        double v = 0;
        switch (b.info.type) {
        case DataType::Float32: v = b.as<float>()[i]; break;
        case DataType::UInt8: v = b.as<uint8_t>()[i]; break;
        case DataType::Int8: v = b.as<int8_t>()[i]; break;
        case DataType::Int32: v = b.as<int32_t>()[i]; break;
        case DataType::Int64: v = static_cast<double>(b.as<int64_t>()[i]); break;
        default: v = 0;
        }
        if (b.info.scale != 0 && b.info.type != DataType::Float32)
            v = b.info.scale * (v - static_cast<double>(b.info.zero_point));
        std::printf(" %.4g", v);
    }
    std::printf("%s\n", element_count(b.info.shape) > n ? " ..." : "");
}

int run(int argc, char **argv, int i, bool session_bus)
{
    if (i >= argc)
        return usage();
    InferenceBuilder builder;
    builder.model(argv[i++]).session_bus(session_bus);
    std::string input;
    int repeat = 1, priority = 0;
    bool async = false;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) input = argv[++i];
        else if (a == "--repeat" && i + 1 < argc) repeat = std::max(1, std::atoi(argv[++i]));
        else if (a == "--priority" && i + 1 < argc) priority = std::atoi(argv[++i]);
        else if (a == "--async") async = true;
        else if (a == "--local") builder.mode(Mode::Local);
        else if (a == "--config" && i + 1 < argc) builder.config(argv[++i]);
        else return usage();
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<Session> s = builder.build();
    const double setup = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%s: %s, ready in %.1f ms\n", s->model().c_str(), s->remote() ? "esca-inferenced" : "in-process",
                setup);

    // Input: the file's bytes, or a fixed ramp so runs are comparable.
    for (size_t k = 0; k < s->inputs().size(); ++k) {
        TensorBuffer &b = s->inputs()[k];
        auto *p = static_cast<uint8_t *>(b.data);
        if (!input.empty() && k == 0) {
            std::ifstream f(input, std::ios::binary);
            f.read(reinterpret_cast<char *>(p), static_cast<std::streamsize>(b.info.bytes));
            std::printf("input %zu: %lld bytes from %s\n", k, static_cast<long long>(f.gcount()), input.c_str());
        } else {
            for (size_t j = 0; j < b.info.bytes; ++j)
                p[j] = static_cast<uint8_t>(j * 7);
        }
    }

    std::vector<double> run_ms;
    std::string backend;
    for (int r = 0; r < repeat; ++r) {
        InferenceTiming t;
        if (async) {
            std::atomic<bool> done { false };
            ErrorCode code = ErrorCode::Ok;
            std::string message;
            s->run_async(
                [&](ErrorCode c, const std::string &m, const InferenceTiming &timing) {
                    code = c;
                    message = m;
                    t = timing;
                    done = true;
                },
                priority);
            while (!done)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (code != ErrorCode::Ok)
                raise(code, message);
        } else {
            t = s->run(priority);
        }
        run_ms.push_back(t.run_ms);
        backend = to_string(t.backend);
    }
    std::sort(run_ms.begin(), run_ms.end());
    double sum = 0;
    for (double v : run_ms)
        sum += v;
    std::printf("%d run(s) on %s: avg %.1f ms, min %.1f, median %.1f, max %.1f\n", repeat, backend.c_str(),
                sum / repeat, run_ms.front(), run_ms[run_ms.size() / 2], run_ms.back());
    std::printf("outputs:\n");
    for (const TensorBuffer &b : s->outputs())
        print_values(b);
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    int i = 1;
    bool session_bus = false;
    if (i < argc && std::string(argv[i]) == "--session") {
        session_bus = true;
        ++i;
    }
    if (i >= argc)
        return usage();
    const std::string cmd = argv[i++];

    try {
        if (cmd == "run")
            return run(argc, argv, i, session_bus);

        InferenceServiceClient c(session_bus);
        if (cmd == "list") {
            std::printf("%-20s %-11s %-24s %-12s %8s  %s\n", "MODEL", "STATE", "BACKENDS", "LOADED ON", "KIB", "PATH");
            for (const ModelSummaryInfo &m : c.list_models()) {
                std::printf("%-20s %-11s %-24s %-12s %8llu  %s\n", m.name.c_str(), m.state.c_str(),
                            m.preferred.c_str(), m.loaded.empty() ? "-" : m.loaded.c_str(),
                            static_cast<unsigned long long>(m.bytes / 1024), m.path.c_str());
                if (!m.last_error.empty())
                    std::printf("%-20s last error: %s\n", "", m.last_error.c_str());
            }
        } else if (cmd == "info" && i < argc) {
            const auto [in, out] = c.model_info(argv[i]);
            print_tensors("inputs", in);
            print_tensors("outputs", out);
        } else if (cmd == "config" && i < argc) {
            std::printf("%s\n", c.model_config(argv[i]).c_str());
        } else if (cmd == "register" && i < argc) {
            std::ifstream f(argv[i]);
            std::stringstream text;
            text << f.rdbuf();
            std::printf("registered %s\n", c.register_model(text.str()).c_str());
        } else if (cmd == "unregister" && i < argc) {
            c.unregister_model(argv[i]);
        } else if (cmd == "load" && i < argc) {
            std::printf("%s loaded on %s\n", argv[i], c.load_model(argv[i]).c_str());
        } else if (cmd == "unload" && i < argc) {
            c.unload_model(argv[i]);
        } else if (cmd == "refresh") {
            const auto [registered, errors] = c.refresh();
            for (const std::string &n : registered)
                std::printf("registered %s\n", n.c_str());
            for (const std::string &e : errors)
                std::printf("error: %s\n", e.c_str());
        } else if (cmd == "stats") {
            const ServiceStats s = c.stats();
            std::printf("queued %u, running %u, completed %llu, failed %llu\n", s.queued, s.running,
                        static_cast<unsigned long long>(s.completed), static_cast<unsigned long long>(s.failed));
            std::printf("latency avg %.1f ms, p95 %.1f ms, queue wait avg %.1f ms\n", s.avg_ms, s.p95_ms,
                        s.avg_wait_ms);
            std::printf("scheduling %s, eviction %s, %u context(s), %llu KiB of models loaded\n",
                        s.scheduling.c_str(), s.eviction.c_str(), s.contexts,
                        static_cast<unsigned long long>(s.loaded_bytes / 1024));
        } else if (cmd == "policy" && i + 1 < argc) {
            const std::string which = argv[i];
            if (which == "scheduling") c.set_scheduling_policy(argv[i + 1]);
            else if (which == "eviction") c.set_eviction_policy(argv[i + 1]);
            else return usage();
        } else {
            return usage();
        }
        return 0;
    } catch (const InferenceException &e) {
        std::fprintf(stderr, "esca-inferctl: %s (%s)\n", e.what(), to_string(e.code()));
        return 1;
    }
}
