// Just enough of a test harness: named cases, CHECK, and a failure count
// for the exit status. Kept dependency-free so the tests build anywhere the
// code does, on the host or in the image.

#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace check {

struct Case {
    const char *name;
    std::function<void()> body;
};

inline std::vector<Case> &cases()
{
    static std::vector<Case> all;
    return all;
}

inline int &failures()
{
    static int n = 0;
    return n;
}

struct Register {
    Register(const char *name, std::function<void()> body) { cases().push_back({ name, std::move(body) }); }
};

inline int run_all()
{
    int failed_cases = 0;
    for (const Case &c : cases()) {
        const int before = failures();
        try {
            c.body();
        } catch (const std::exception &e) {
            std::printf("  unexpected exception: %s\n", e.what());
            ++failures();
        }
        const bool ok = failures() == before;
        failed_cases += ok ? 0 : 1;
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", c.name);
    }
    std::printf("%zu cases, %d failed\n", cases().size(), failed_cases);
    return failed_cases == 0 ? 0 : 1;
}

} // namespace check

#define CHECK_CAT2(a, b) a##b
#define CHECK_CAT(a, b) CHECK_CAT2(a, b)
#define TEST(name)                                                                                   \
    static void CHECK_CAT(test_, __LINE__)();                                                        \
    static check::Register CHECK_CAT(reg_, __LINE__)(name, CHECK_CAT(test_, __LINE__));             \
    static void CHECK_CAT(test_, __LINE__)()

#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);                   \
            ++check::failures();                                                                     \
        }                                                                                            \
    } while (0)

/// Expects `expr` to throw an InferenceException-like type with `code()`.
#define CHECK_THROWS_CODE(expr, expected)                                                            \
    do {                                                                                             \
        bool thrown = false;                                                                         \
        try {                                                                                        \
            expr;                                                                                    \
        } catch (const esca::inference::InferenceException &e) {                                    \
            thrown = true;                                                                           \
            if (e.code() != (expected)) {                                                            \
                std::printf("  %s:%d: threw %s, expected %s\n", __FILE__, __LINE__,                 \
                            esca::inference::to_string(e.code()),                                    \
                            esca::inference::to_string(expected));                                   \
                ++check::failures();                                                                 \
            }                                                                                        \
        }                                                                                            \
        if (!thrown) {                                                                               \
            std::printf("  %s:%d: %s did not throw\n", __FILE__, __LINE__, #expr);                   \
            ++check::failures();                                                                     \
        }                                                                                            \
    } while (0)
