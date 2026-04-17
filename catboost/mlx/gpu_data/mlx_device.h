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

        // Explicit sync at well-defined compute boundaries.
        // Use at points where a CPU readback (e.g. .data<T>()) follows immediately,
        // or at iteration loop boundaries to bound lazy-graph depth.
        // Use sparingly — each call blocks until the GPU drains the command buffer.
        static void EvalAtBoundary(const mx::array& arr) {
            mx::eval(arr);
        }

        static void EvalAtBoundary(const std::vector<mx::array>& arrays) {
            mx::eval(arrays);
        }

        // DEPRECATED: use EvalAtBoundary at explicit sync boundaries.
        // EvalNow was historically called mid-computation, creating unnecessary
        // CPU-GPU sync barriers. All new code must use EvalAtBoundary instead.
        [[deprecated("Use EvalAtBoundary at explicit sync boundaries")]]
        static void EvalNow(const mx::array& arr) {
            mx::eval(arr);
        }

        [[deprecated("Use EvalAtBoundary at explicit sync boundaries")]]
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
