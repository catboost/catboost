#include <catboost/cuda/gpu_data/kernel/gpu_input_factorize.cuh>

#include <catboost/libs/cat_feature/cat_feature.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/vector.h>
#include <util/string/cast.h>

#include <cuda_runtime.h>

#include <limits>

namespace {

    template <class TValue, class TStringValue>
    static ui32 CalcCpuCatHash(const TValue v) {
        return CalcCatFeatureHash(ToString(static_cast<TStringValue>(v)));
    }

    template <class TValue, class TStringValue>
    static void TestGpuCatHash(NKernel::EGpuInputDType dtype, const TVector<TValue>& values) {
        UNIT_ASSERT(!values.empty());

        CUDA_SAFE_CALL(cudaSetDevice(0));

        TValue* dValues = nullptr;
        ui32* dHashes = nullptr;
        CUDA_SAFE_CALL(cudaMalloc(&dValues, values.size() * sizeof(TValue)));
        CUDA_SAFE_CALL(cudaMalloc(&dHashes, values.size() * sizeof(ui32)));
        Y_DEFER {
            if (dValues) {
                cudaFree(dValues);
            }
            if (dHashes) {
                cudaFree(dHashes);
            }
        };

        CUDA_SAFE_CALL(cudaMemcpy(dValues, values.data(), values.size() * sizeof(TValue), cudaMemcpyHostToDevice));

        NKernel::HashUniqueNumericToCatHash(
            dValues,
            static_cast<ui32>(values.size()),
            dtype,
            dHashes,
            /*stream*/ 0
        );
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        TVector<ui32> hashes;
        hashes.yresize(values.size());
        CUDA_SAFE_CALL(cudaMemcpy(hashes.data(), dHashes, hashes.size() * sizeof(ui32), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < values.size(); ++i) {
            const ui32 expected = CalcCpuCatHash<TValue, TStringValue>(values[i]);
            UNIT_ASSERT_VALUES_EQUAL_C(hashes[i], expected, "index=" << i);
        }
    }

}

Y_UNIT_TEST_SUITE(GpuInputCatHashTests) {
    Y_UNIT_TEST(Int8) {
        TestGpuCatHash<i8, i64>(NKernel::EGpuInputDType::Int8, {-128, -1, 0, 1, 2, 10, 127});
    }

    Y_UNIT_TEST(Int16) {
        TestGpuCatHash<i16, i64>(NKernel::EGpuInputDType::Int16, {-32768, -12345, -1, 0, 1, 12345, 32767});
    }

    Y_UNIT_TEST(Int32) {
        TestGpuCatHash<i32, i64>(NKernel::EGpuInputDType::Int32, {-(1 << 30), -123456789, -1, 0, 1, 123456789, (1 << 30)});
    }

    Y_UNIT_TEST(Int64) {
        TestGpuCatHash<i64, i64>(
            NKernel::EGpuInputDType::Int64,
            {
                std::numeric_limits<i64>::min(),
                std::numeric_limits<i64>::min() + 1,
                -1234567890123456789LL,
                -1,
                0,
                1,
                1234567890123456789LL,
                std::numeric_limits<i64>::max()
            }
        );
    }

    Y_UNIT_TEST(UInt8) {
        TestGpuCatHash<ui8, ui64>(NKernel::EGpuInputDType::UInt8, {0, 1, 2, 10, 127, 255});
    }

    Y_UNIT_TEST(UInt16) {
        TestGpuCatHash<ui16, ui64>(NKernel::EGpuInputDType::UInt16, {0, 1, 2, 10, 32767, 65535});
    }

    Y_UNIT_TEST(UInt32) {
        TestGpuCatHash<ui32, ui64>(NKernel::EGpuInputDType::UInt32, {0u, 1u, 2u, 10u, 123456789u, 4294967295u});
    }

    Y_UNIT_TEST(UInt64) {
        TestGpuCatHash<ui64, ui64>(
            NKernel::EGpuInputDType::UInt64,
            {
                0ULL,
                1ULL,
                2ULL,
                10ULL,
                1234567890123456789ULL,
                std::numeric_limits<ui64>::max()
            }
        );
    }
}
