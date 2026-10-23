// TensorFlow Lite, on the CPU or through the XNNPACK delegate.
//
// TFLite stays behind a pimpl, as in esca-visiond: its interpreter type is an
// alias into an inner namespace and cannot be forward declared, and nothing
// above this file should compile against TensorFlow headers.

#pragma once

#include <memory>
#include <string>

#include "strategy.hpp"

namespace esca::inference {

class TfliteStrategy : public InferenceStrategy {
public:
    TfliteStrategy(std::string model_path, Backend backend, int threads);
    ~TfliteStrategy() override;

    Backend backend() const override { return backend_; }
    void load() override;
    void unload() override;
    bool loaded() const override;
    std::vector<TensorInfo> inputs() const override;
    std::vector<TensorInfo> outputs() const override;
    void resize_inputs(const std::vector<std::vector<int32_t>> &shapes) override;
    void run(const std::vector<ConstBuffer> &inputs,
             const std::vector<MutableBuffer> &outputs) override;
    size_t memory_footprint() const override;

    static bool supports(Backend backend);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string path_;
    Backend backend_;
    int threads_;
};

} // namespace esca::inference
