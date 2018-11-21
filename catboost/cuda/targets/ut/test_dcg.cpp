#include <library/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/targets/dcg.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/random/fast.h>
#include <util/stream/labeled.h>

using NCatboostCuda::NDetail::MakeDcgDecay;
using NCatboostCuda::NDetail::MakeDcgExponentialDecay;
using NCudaLib::GetCudaManager;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;

static TVector<ui32> MakeOffsets(const ui32 groupCount, const ui32 maxGroupSize, const ui64 seed) {
    TFastRng<ui64> prng(seed);
    TVector<ui32> offsets;
    offsets.reserve(static_cast<size_t>(groupCount) * maxGroupSize);

    for (ui32 offset = 0, group = 0; group < groupCount; ++group) {
        offset = offsets.size();
        const ui32 groupSize = prng.Uniform(maxGroupSize) + 1;
        for (ui32 i = 0; i < groupSize; ++i) {
            offsets.push_back(offset);
        }
    }

    return offsets;
}

static TStripeMapping MakeGroupAwareStripeMapping(const TConstArrayRef<ui32> offsets) {
    // We don't want group to be split across multiple devies
    const auto deviceCount = GetCudaManager().GetDeviceCount();
    const auto objectsPerDevice = (offsets.size() + deviceCount - 1) / deviceCount;
    TVector<TSlice> slices;
    slices.reserve(deviceCount);

    size_t start = 0;
    while (start < offsets.size()) {
        auto end = Min(start + objectsPerDevice, offsets.size());
        for (; end < offsets.size() && offsets[end - 1] == offsets[end]; ++end) {
        }
        slices.emplace_back(start, end);
        start = end;
    }

    // init rest of slices
    slices.resize(deviceCount);
    return TStripeMapping(std::move(slices));
}

static TVector<ui32> MakeBiasedOffsets(const TConstArrayRef<ui32> offsets, const TStripeMapping& mapping) {
    const auto deviceCount = GetCudaManager().GetDeviceCount();
    TVector<ui32> biasedOffsets;
    biasedOffsets.yresize(offsets.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        for (ui64 i = slice.Left; i < slice.Right; ++i) {
            biasedOffsets[i] = offsets[i] - slice.Left;
        }
    }
    return biasedOffsets;
}

static TVector<float> MakeDcgDecay(const TConstArrayRef<ui32> offsets) {
    TVector<float> decay;
    decay.yresize(offsets.size());
    for (ui32 i = 0, iEnd = offsets.size(); i < iEnd; ++i) {
        decay[i] = 1.f / Log2(static_cast<float>(i - offsets[i] + 2));
    }
    return decay;
}

static TVector<float> MakeDcgExponentialDecay(const TConstArrayRef<ui32> offsets, const float base) {
    TVector<float> decay;
    decay.yresize(offsets.size());
    for (ui32 i = 0, iEnd = offsets.size(); i < iEnd; ++i) {
        decay[i] = pow(base, i - offsets[i]);
    }
    return decay;
}

Y_UNIT_TEST_SUITE(NdcgTests) {
    Y_UNIT_TEST(TestMakeDcgDecaySingleDevice) {
        const auto devicesGuard = StartCudaManager();
        const ui32 groupCount = 1000000;
        const ui32 maxGroupSize = 30;
        const ui64 seed = 0;

        const auto offsets = MakeOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgDecay(offsets);

        const TSingleMapping mapping(0, offsets.size());
        auto deviceDecay = TSingleBuffer<float>::Create(mapping);
        auto deviceOffsets = TSingleBuffer<ui32>::Create(mapping);

        deviceOffsets.Write(offsets);
        MakeDcgDecay(deviceOffsets, deviceDecay);

        TVector<float> gpuDecay;
        deviceDecay.Read(gpuDecay);

        UNIT_ASSERT_VALUES_EQUAL(cpuDecay.size(), gpuDecay.size());
        for (size_t i = 0, iEnd = cpuDecay.size(); i < iEnd; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(
                cpuDecay[i], gpuDecay[i], 1e-5,
                LabeledOutput(i, offsets[i], i - offsets[i] + 2));
        }
    }

    Y_UNIT_TEST(TestMakeDcgDecayMultipleDevices) {
        const auto devicesGuard = StartCudaManager();
        const ui32 groupCount = 1000000;
        const ui32 maxGroupSize = 30;
        const ui64 seed = 0;

        const auto offsets = MakeOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgDecay(offsets);

        const auto mapping = MakeGroupAwareStripeMapping(offsets);
        auto deviceDecay = TStripeBuffer<float>::Create(mapping);
        auto deviceOffsets = TStripeBuffer<ui32>::Create(mapping);

        deviceOffsets.Write(MakeBiasedOffsets(offsets, mapping));
        MakeDcgDecay(deviceOffsets, deviceDecay);

        TVector<float> gpuDecay;
        deviceDecay.Read(gpuDecay);

        UNIT_ASSERT_VALUES_EQUAL(cpuDecay.size(), gpuDecay.size());
        for (size_t i = 0, iEnd = cpuDecay.size(); i < iEnd; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(
                cpuDecay[i], gpuDecay[i], 1e-5,
                LabeledOutput(i, offsets[i], i - offsets[i] + 2));
        }
    }

    Y_UNIT_TEST(TestMakeDcgExponentialDecaySingleDevice) {
        const auto devicesGuard = StartCudaManager();
        const ui32 groupCount = 1000000;
        const ui32 maxGroupSize = 30;
        const ui64 seed = 0;
        const float base = 1.3f;

        const auto offsets = MakeOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgExponentialDecay(offsets, base);

        const TSingleMapping mapping(0, offsets.size());
        auto deviceOffsets = TSingleBuffer<ui32>::Create(mapping);
        auto deviceDecay = TSingleBuffer<float>::Create(mapping);

        deviceOffsets.Write(offsets);
        MakeDcgExponentialDecay(deviceOffsets, base, deviceDecay);

        TVector<float> gpuDecay;
        deviceDecay.Read(gpuDecay);

        UNIT_ASSERT_VALUES_EQUAL(cpuDecay.size(), gpuDecay.size());
        for (size_t i = 0, iEnd = cpuDecay.size(); i < iEnd; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(
                cpuDecay[i], gpuDecay[i], 1e-5,
                LabeledOutput(i, offsets[i], i - offsets[i]));
        }
    }

    Y_UNIT_TEST(TestMakeDcgExponentialDecayMultipleDevices) {
        const auto devicesGuard = StartCudaManager();
        const ui32 groupCount = 1000000;
        const ui32 maxGroupSize = 30;
        const ui64 seed = 0;
        const float base = 1.3f;

        const auto offsets = MakeOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgExponentialDecay(offsets, base);

        const auto mapping = MakeGroupAwareStripeMapping(offsets);
        auto deviceDecay = TStripeBuffer<float>::Create(mapping);
        auto deviceOffsets = TStripeBuffer<ui32>::Create(mapping);

        deviceOffsets.Write(MakeBiasedOffsets(offsets, mapping));
        MakeDcgExponentialDecay(deviceOffsets, base, deviceDecay);

        TVector<float> gpuDecay;
        deviceDecay.Read(gpuDecay);

        UNIT_ASSERT_VALUES_EQUAL(cpuDecay.size(), gpuDecay.size());
        for (size_t i = 0, iEnd = cpuDecay.size(); i < iEnd; ++i) {
            UNIT_ASSERT_DOUBLES_EQUAL_C(
                cpuDecay[i], gpuDecay[i], 1e-5,
                LabeledOutput(i, offsets[i], i - offsets[i]));
        }
    }
}
