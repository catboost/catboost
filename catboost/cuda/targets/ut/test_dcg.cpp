#include <library/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/targets/dcg.h>
#include <catboost/libs/helpers/exception.h>

#include <library/float16/float16.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/random/fast.h>
#include <util/stream/labeled.h>

using NCatboostCuda::NDetail::FuseUi32AndFloatIntoUi64;
using NCatboostCuda::NDetail::FuseUi32AndTwoFloatsIntoUi64;
using NCatboostCuda::NDetail::GetBits;
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

static TVector<ui64> FuseUi32AndFloatIntoUi64(
    const TConstArrayRef<ui32> ui32s,
    const TConstArrayRef<float> floats)
{
    TVector<ui64> fused;
    fused.yresize(ui32s.size());

    for (size_t i = 0; i < ui32s.size(); ++i) {
        const auto casted = *reinterpret_cast<const ui32*>(&floats[i]);
        const ui32 mask = -i32(casted >> 31) | (i32(1) << 31);
        fused[i] = (static_cast<ui64>(ui32s[i]) << 32) | (casted ^ mask);
    }

    return fused;
}

static TVector<ui64> FuseUi32AndTwoFloatsIntoUi64(
    const TConstArrayRef<ui32> ui32s,
    const TConstArrayRef<float> floats1,
    const TConstArrayRef<float> floats2)
{
    TVector<ui64> fused;
    fused.yresize(ui32s.size());

    for (size_t i = 0; i < ui32s.size(); ++i) {
        ui64 value = static_cast<ui64>(ui32s[i]) << 32;
        {
            const auto casted = TFloat16(floats1[i]).Data;
            const ui16 mask = -i16(casted >> 15) | (i16(1) << 15);
            value |= static_cast<ui64>(casted ^ mask) << 16;
        }
        {
            const auto casted = TFloat16(floats2[i]).Data;
            const ui16 mask = -i16(casted >> 15) | (i16(1) << 15);
            value |= static_cast<ui64>(casted ^ mask);
        }
        fused[i] = value;
    }

    return fused;
}

template <typename T, typename U>
static TVector<U> GetBits(const TConstArrayRef<T> src, const ui32 bitsOffset, const ui32 bitsCount)
{
    CB_ENSURE(bitsCount <= sizeof(T) * 8, LabeledOutput(bitsCount, sizeof(T) * 8));
    CB_ENSURE(bitsCount <= sizeof(U) * 8, LabeledOutput(bitsCount, sizeof(U) * 8));
    CB_ENSURE(bitsOffset <= sizeof(T) * 8, LabeledOutput(bitsOffset, sizeof(T) * 8));

    TVector<U> dst;
    dst.yresize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = (src[i] << (sizeof(T) - bitsOffset + bitsCount)) >> (sizeof(T) + bitsCount);
    }
    return dst;
}

static void Sort(
    const TConstArrayRef<ui32> ui32s,
    const TConstArrayRef<float> floats,
    const TStripeMapping& mapping,
    TVector<ui32>* const sortedUi32s,
    TVector<float>* const sortedFloats)
{
    const auto deviceCount = GetCudaManager().GetDeviceCount();

    // sort in the same way as documents are sorted in IDCG calculation
    TVector<ui32> indices;
    indices.yresize(ui32s.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        Iota(indices.begin() + slice.Left, indices.begin() + slice.Right, static_cast<ui32>(0));
    }

    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        Sort(
            indices.begin() + slice.Left, indices.begin() + slice.Right,
            [offset = slice.Left, ui32s, floats](const auto lhs, const auto rhs) {
                if (ui32s[offset + lhs] == ui32s[offset + rhs]) {
                    return floats[offset + lhs] > floats[offset + rhs];
                }

                return ui32s[offset + lhs] > ui32s[offset + rhs];
        });
    }

    sortedUi32s->yresize(ui32s.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        for (ui64 i = slice.Left; i < slice.Right; ++i) {
            (*sortedUi32s)[i] = ui32s[slice.Left + indices[i]];
        }
    }

    sortedFloats->yresize(floats.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        for (ui64 i = slice.Left; i < slice.Right; ++i) {
            (*sortedFloats)[i] = floats[slice.Left + indices[i]];
        }
    }
}

static void Sort(
    const TConstArrayRef<ui32> ui32s,
    const TConstArrayRef<float> floats1,
    const TConstArrayRef<float> floats2,
    const TStripeMapping& mapping,
    TVector<ui32>* const sortedUi32s,
    TVector<float>* const sortedFloats1,
    TVector<float>* const sortedFloats2)
{
    const auto deviceCount = GetCudaManager().GetDeviceCount();

    // sort in the same way as documents are sorted in DCG calculation
    TVector<ui32> indices;
    indices.yresize(ui32s.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        Iota(indices.begin() + slice.Left, indices.begin() + slice.Right, static_cast<ui32>(0));
    }

    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        Sort(
            indices.begin() + slice.Left, indices.begin() + slice.Right,
            [offset = slice.Left, ui32s, floats1, floats2](const auto lhs, const auto rhs) {
                if (ui32s[offset + lhs] == ui32s[offset + rhs]) {
                    const TFloat16 lhsFloat1 = floats1[offset + lhs];
                    const TFloat16 rhsFloat1 = floats1[offset + rhs];
                    if (lhsFloat1 == rhsFloat1) {
                        return TFloat16(floats2[offset + lhs]) < TFloat16(floats2[offset + rhs]);
                    }

                    return TFloat16(lhsFloat1) > TFloat16(rhsFloat1);
                }

                return ui32s[offset + lhs] > ui32s[offset + rhs];
        });
    }

    sortedUi32s->yresize(ui32s.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        for (ui64 i = slice.Left; i < slice.Right; ++i) {
            (*sortedUi32s)[i] = ui32s[slice.Left + indices[i]];
        }
    }

    sortedFloats1->yresize(floats1.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        for (ui64 i = slice.Left; i < slice.Right; ++i) {
            (*sortedFloats1)[i] = floats1[slice.Left + indices[i]];
        }
    }

    sortedFloats2->yresize(floats1.size());
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        for (ui64 i = slice.Left; i < slice.Right; ++i) {
            (*sortedFloats2)[i] = floats2[slice.Left + indices[i]];
        }
    }
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

    Y_UNIT_TEST(TestFuseUi32AndFloatIntoUi64) {
        const auto devicesGuard = StartCudaManager();
        const size_t size = 30000000;
        const ui32 groupCount = size / 30;
        const ui64 seed = 0;
        const float scaleFactor = 100.f;

        TFastRng<ui64> prng(seed);
        TVector<ui32> ui32s;
        ui32s.yresize(size);
        TVector<float> floats;
        floats.yresize(size);
        for (size_t i = 0; i < size; ++i) {
            ui32s[i] = prng.Uniform(groupCount);
            const auto positive = (prng.Uniform(2) == 0);
            floats[i] = (positive ? 1.f : -1.f) * prng.GenRandReal4() * scaleFactor;
        }

        const auto cpuFused = FuseUi32AndFloatIntoUi64(ui32s, floats);

        const auto mapping = TStripeMapping::SplitBetweenDevices(size);
        auto deviceUi32s = TStripeBuffer<ui32>::Create(mapping);
        auto deviceFloats = TStripeBuffer<float>::Create(mapping);
        auto deviceFused = TStripeBuffer<ui64>::Create(mapping);

        deviceUi32s.Write(ui32s);
        deviceFloats.Write(floats);
        FuseUi32AndFloatIntoUi64(deviceUi32s, deviceFloats, deviceFused);

        TVector<ui64> gpuFused;
        deviceFused.Read(gpuFused);

        UNIT_ASSERT_VALUES_EQUAL(cpuFused.size(), gpuFused.size());
        for (size_t i = 0, iEnd = cpuFused.size(); i < iEnd; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(
                cpuFused[i], gpuFused[i],
                "at " << i);
        }
    }

    Y_UNIT_TEST(TestFuseUi32AndTwoFloatsIntoUi64) {
        const auto devicesGuard = StartCudaManager();
        const size_t size = 30000000;
        const ui32 groupCount = size / 30;
        const ui64 seed = 0;
        const float scaleFactor = 100.f;

        TFastRng<ui64> prng(seed);
        TVector<ui32> ui32s;
        ui32s.yresize(size);
        TVector<float> floats1;
        floats1.yresize(size);
        TVector<float> floats2;
        floats2.yresize(size);
        for (size_t i = 0; i < size; ++i) {
            ui32s[i] = prng.Uniform(groupCount);
            {
                const auto positive = (prng.Uniform(2) == 0);
                floats1[i] = (positive ? 1.f : -1.f) * prng.GenRandReal4() * scaleFactor;
                if (TFloat16(floats1[i]) == 0.f) {
                    // there is some inconsistency between cpu and gpu implementation of float to
                    // half, just ignore everything that get casted to zero
                    floats1[i] = 0.f;
                }
            }
            {
                const auto positive = (prng.Uniform(2) == 0);
                floats2[i] = (positive ? 1.f : -1.f) * prng.GenRandReal4() * scaleFactor;
                if (TFloat16(floats2[i]) == -0.f) {
                    floats2[i] = 0.f;
                }
            }
        }

        const auto cpuFused = FuseUi32AndTwoFloatsIntoUi64(ui32s, floats1, floats2);

        const auto mapping = TStripeMapping::SplitBetweenDevices(size);
        auto deviceUi32s = TStripeBuffer<ui32>::Create(mapping);
        auto deviceFloats1 = TStripeBuffer<float>::Create(mapping);
        auto deviceFloats2 = TStripeBuffer<float>::Create(mapping);
        auto deviceFused = TStripeBuffer<ui64>::Create(mapping);

        deviceUi32s.Write(ui32s);
        deviceFloats1.Write(floats1);
        deviceFloats2.Write(floats2);
        FuseUi32AndTwoFloatsIntoUi64(deviceUi32s, deviceFloats1, deviceFloats2, deviceFused);

        TVector<ui64> gpuFused;
        deviceFused.Read(gpuFused);

        UNIT_ASSERT_VALUES_EQUAL(cpuFused.size(), gpuFused.size());
        for (size_t i = 0, iEnd = cpuFused.size(); i < iEnd; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(
                cpuFused[i], gpuFused[i],
                LabeledOutput(i, ui32s[i], floats1[i], floats2[i], TFloat16(floats1[i]), TFloat16(floats2[i])));
        }
    }

    Y_UNIT_TEST(TestSortFusedUi32AndFloat) {
        const auto devicesGuard = StartCudaManager();
        const size_t size = 30000000;
        const ui32 groupCount = size / 30;
        const ui64 seed = 0;
        const float scaleFactor = 100.f;

        TFastRng<ui64> prng(seed);
        TVector<ui32> ui32s;
        ui32s.yresize(size);
        TVector<float> floats;
        floats.yresize(size);
        for (size_t i = 0; i < size; ++i) {
            ui32s[i] = prng.Uniform(groupCount);
            const auto positive = (prng.Uniform(2) == 0);
            floats[i] = (positive ? 1.f : -1.f) * prng.GenRandReal4() * scaleFactor;
        }

        const auto mapping = TStripeMapping::SplitBetweenDevices(size);

        TVector<ui32> cpuSortedUi32s;
        TVector<float> cpuSortedFloats;
        Sort(ui32s, floats, mapping, &cpuSortedUi32s, &cpuSortedFloats);

        auto deviceUi32s = TStripeBuffer<ui32>::Create(mapping);
        auto deviceSortedUi32s = TStripeBuffer<ui32>::Create(mapping);
        auto deviceFloats = TStripeBuffer<float>::Create(mapping);
        auto deviceSortedFloats = TStripeBuffer<float>::Create(mapping);
        auto deviceFused = TStripeBuffer<ui64>::Create(mapping);
        auto deviceIndices = TStripeBuffer<ui32>::Create(mapping);

        deviceUi32s.Write(ui32s);
        deviceFloats.Write(floats);

        FuseUi32AndFloatIntoUi64(deviceUi32s, deviceFloats, deviceFused);
        MakeSequence(deviceIndices);
        RadixSort(deviceFused, deviceIndices, true);
        Gather(deviceSortedUi32s, deviceUi32s, deviceIndices);
        Gather(deviceSortedFloats, deviceFloats, deviceIndices);

        TVector<ui32> gpuSortedUi32s;
        TVector<float> gpuSortedFloats;
        deviceSortedUi32s.Read(gpuSortedUi32s);
        deviceSortedFloats.Read(gpuSortedFloats);

        UNIT_ASSERT_VALUES_EQUAL(cpuSortedUi32s.size(), gpuSortedUi32s.size());
        UNIT_ASSERT_VALUES_EQUAL(cpuSortedFloats.size(), gpuSortedFloats.size());
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(cpuSortedUi32s[i], gpuSortedUi32s[i], "at " << i);
            UNIT_ASSERT_VALUES_EQUAL_C(cpuSortedFloats[i], gpuSortedFloats[i], "at " << i);
        }
    }

    Y_UNIT_TEST(TestSortFusedUi32AndTwoFloats) {
        const auto devicesGuard = StartCudaManager();
        const size_t size = 30000000;
        // const size_t size = 30;
        const ui32 groupCount = size / 30;
        const ui64 seed = 0;
        const float scaleFactor = 100.f;

        TFastRng<ui64> prng(seed);
        TVector<ui32> ui32s;
        ui32s.yresize(size);
        TVector<float> floats1;
        floats1.yresize(size);
        TVector<float> floats2;
        floats2.yresize(size);
        for (size_t i = 0; i < size; ++i) {
            ui32s[i] = prng.Uniform(groupCount);
            {
                const auto positive = (prng.Uniform(2) == 0);
                floats1[i] = (positive ? 1.f : -1.f) * prng.GenRandReal4() * scaleFactor;
                if (TFloat16(floats1[i]) == -0.f) {
                    // there is some inconsistency between cpu and gpu implementation of float to
                    // half, just ignore everything that get casted to zero
                    floats1[i] = 0.f;
                }
            }
            {
                const auto positive = (prng.Uniform(2) == 0);
                floats2[i] = (positive ? 1.f : -1.f) * prng.GenRandReal4() * scaleFactor;
                if (TFloat16(floats2[i]) == -0.f) {
                    floats2[i] = 0.f;
                }
            }
        }

        const auto mapping = TStripeMapping::SplitBetweenDevices(size);

        TVector<ui32> cpuSortedUi32s;
        TVector<float> cpuSortedFloats1;
        TVector<float> cpuSortedFloats2;
        Sort(ui32s, floats1, floats2, mapping, &cpuSortedUi32s, &cpuSortedFloats1, &cpuSortedFloats2);

        auto deviceUi32s = TStripeBuffer<ui32>::Create(mapping);
        auto deviceSortedUi32s = TStripeBuffer<ui32>::Create(mapping);
        auto deviceFloats1 = TStripeBuffer<float>::Create(mapping);
        auto deviceSortedFloats1 = TStripeBuffer<float>::Create(mapping);
        auto deviceFloats2 = TStripeBuffer<float>::Create(mapping);
        auto deviceSortedFloats2 = TStripeBuffer<float>::Create(mapping);
        auto deviceFused = TStripeBuffer<ui64>::Create(mapping);
        auto deviceIndices = TStripeBuffer<ui32>::Create(mapping);

        deviceUi32s.Write(ui32s);
        deviceFloats1.Write(floats1);
        deviceFloats2.Write(floats2);

        FuseUi32AndTwoFloatsIntoUi64(deviceUi32s, deviceFloats1, deviceFloats2, deviceFused, false, true);
        MakeSequence(deviceIndices);
        RadixSort(deviceFused, deviceIndices, true);
        Gather(deviceSortedUi32s, deviceUi32s, deviceIndices);
        Gather(deviceSortedFloats1, deviceFloats1, deviceIndices);
        Gather(deviceSortedFloats2, deviceFloats2, deviceIndices);

        TVector<ui32> gpuSortedUi32s;
        TVector<float> gpuSortedFloats1;
        TVector<float> gpuSortedFloats2;
        deviceSortedUi32s.Read(gpuSortedUi32s);
        deviceSortedFloats1.Read(gpuSortedFloats1);
        deviceSortedFloats2.Read(gpuSortedFloats2);

        UNIT_ASSERT_VALUES_EQUAL(cpuSortedUi32s.size(), gpuSortedUi32s.size());
        UNIT_ASSERT_VALUES_EQUAL(cpuSortedFloats1.size(), gpuSortedFloats1.size());
        UNIT_ASSERT_VALUES_EQUAL(cpuSortedFloats2.size(), gpuSortedFloats2.size());
        for (size_t i = 0; i < size; ++i) {
            const auto cpuSortedUi32 = cpuSortedUi32s[i];
            const auto gpuSortedUi32 = gpuSortedUi32s[i];
            const TFloat16 cpuSortedFloat1 = cpuSortedFloats1[i];
            const TFloat16 cpuSortedFloat2 = cpuSortedFloats2[i];
            const TFloat16 gpuSortedFloat1 = gpuSortedFloats1[i];
            const TFloat16 gpuSortedFloat2 = gpuSortedFloats2[i];
            UNIT_ASSERT_VALUES_EQUAL_C(
                cpuSortedUi32, gpuSortedUi32,
                LabeledOutput(i, cpuSortedFloat1, gpuSortedFloat1, cpuSortedFloat2, gpuSortedFloat2));

            UNIT_ASSERT_VALUES_EQUAL_C(
                cpuSortedFloat1, gpuSortedFloat1,
                LabeledOutput(i, cpuSortedUi32, gpuSortedUi32, cpuSortedFloat2, gpuSortedFloat2));

            UNIT_ASSERT_VALUES_EQUAL_C(
                cpuSortedFloat2, gpuSortedFloat2,
                LabeledOutput(i, cpuSortedUi32, gpuSortedUi32, cpuSortedFloat1, gpuSortedFloat1));
        }
    }

    Y_UNIT_TEST(TestGetBits) {
        const auto devicesGuard = StartCudaManager();
        const size_t size = 30000000;
        const ui64 seed = 0;

        TFastRng<ui64> prng(seed);
        TVector<ui64> src;
        src.yresize(size);
        for (auto& value : src) {
            value = prng();
        }

        const auto mapping = TStripeMapping::SplitBetweenDevices(size);
        auto deviceSrc = TStripeBuffer<ui64>::Create(mapping);
        auto deviceDst = TStripeBuffer<ui32>::Create(mapping);

        deviceSrc.Write(src);

        const std::pair<ui32, ui32> cases[] = {{32, 32}, {3, 5}, {0, 32}};
        for (const auto [bitsOffset, bitsCount] : cases) {
            const auto cpuDst = GetBits<ui64, ui32>(src, bitsOffset, bitsCount);
            GetBits(deviceSrc, deviceDst, bitsOffset, bitsCount);

            TVector<ui32> gpuDst;
            deviceDst.Read(gpuDst);

            UNIT_ASSERT_VALUES_EQUAL(cpuDst.size(), gpuDst.size());
            for (size_t i = 0, iEnd = cpuDst.size(); i < iEnd; ++i) {
                UNIT_ASSERT_VALUES_EQUAL_C(
                    cpuDst[i], gpuDst[i],
                    LabeledOutput(bitsOffset, bitsCount, i));
            }
        }
    }
}
