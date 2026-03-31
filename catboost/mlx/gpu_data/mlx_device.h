#pragma once

// Thin wrapper around MLX's Metal device for CatBoost-MLX.
// Provides device access, stream management, and synchronization points.

#include <mlx/mlx.h>

namespace NCatboostMlx {

    namespace mx = mlx::core;

    class TMLXDevice {
    public:
        TMLXDevice()
            : Device_(mx::Device::gpu)
        {
        }

        // Get the MLX GPU device
        const mx::Device& GetDevice() const {
            return Device_;
        }

        // Get the default stream for this device
        mx::Stream GetStream() const {
            return mx::default_stream(Device_);
        }

        // Force evaluation of all pending lazy operations.
        // Call this at iteration boundaries to ensure results are materialized.
        static void EvalNow(const mx::array& arr) {
            mx::eval(arr);
        }

        static void EvalNow(const std::vector<mx::array>& arrays) {
            mx::eval(arrays);
        }

        // Create a GPU array from raw CPU data
        template <typename T>
        static mx::array FromCPU(const T* data, const mx::Shape& shape, mx::Dtype dtype) {
            return mx::array(data, shape, dtype);
        }

        // Create a zero-initialized GPU array
        static mx::array Zeros(const mx::Shape& shape, mx::Dtype dtype) {
            return mx::zeros(shape, dtype);
        }

        // Create an array of ones
        static mx::array Ones(const mx::Shape& shape, mx::Dtype dtype) {
            return mx::ones(shape, dtype);
        }

    private:
        mx::Device Device_;
    };

}  // namespace NCatboostMlx
