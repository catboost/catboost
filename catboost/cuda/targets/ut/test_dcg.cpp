#include <library/cpp/testing/unittest/registar.h>

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/targets/dcg.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/metrics/dcg.h>
#include <catboost/libs/metrics/sample.h>

#include <library/cpp/accurate_accumulate/accurate_accumulate.h>
#include <library/cpp/float16/float16.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/random/fast.h>
#include <util/stream/labeled.h>

using NCatboostCuda::CalculateDcg;
using NCatboostCuda::CalculateIdcg;
using NCatboostCuda::CalculateNdcg;
using NCatboostCuda::NDetail::FuseUi32AndFloatIntoUi64;
using NCatboostCuda::NDetail::FuseUi32AndTwoFloatsIntoUi64;
using NCatboostCuda::NDetail::MakeDcgDecays;
using NCatboostCuda::NDetail::MakeDcgExponentialDecays;
using NCatboostCuda::NDetail::MakeElementwiseOffsets;
using NCatboostCuda::NDetail::MakeEndOfGroupMarkers;
using NCudaLib::GetCudaManager;
using NCudaLib::TDistributedObject;
using NCudaLib::TSingleMapping;
using NCudaLib::TStripeMapping;
using NMetrics::TSample;

static TVector<ui32> MakeElementwiseOffsets(const ui32 groupCount, const ui32 maxGroupSize, const ui64 seed) {
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

static TStripeMapping MakeGroupAwareStripeMappingFromElementwiseOffsets(const TConstArrayRef<ui32> offsets) {
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
    slices.insert(slices.begin(), deviceCount - slices.size(), TSlice());
    return TStripeMapping(std::move(slices));
}

static TDistributedObject<ui32> MakeOffsetsBias(
    const TConstArrayRef<ui32> biasedOffsets,
    const TStripeMapping& mapping) {
    const auto deviceCount = GetCudaManager().GetDeviceCount();
    auto offsetsBias = GetCudaManager().CreateDistributedObject<ui32>();
    for (ui64 device = 0; device < deviceCount; ++device) {
        const auto slice = mapping.DeviceSlice(device);
        offsetsBias.Set(device, biasedOffsets[slice.Left]);
    }

    return offsetsBias;
}

static TVector<ui32> MakeDeviceLocalElementwiseOffsets(
    const TConstArrayRef<ui32> offsets,
    const TStripeMapping& mapping) {
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

static TStripeMapping MakeGroupAwareStripeMappingFromSizes(const TConstArrayRef<ui32> sizes) {
    const auto deviceCount = GetCudaManager().GetDeviceCount();
    const auto elementsCount = Accumulate(sizes.begin(), sizes.end(), size_t(0));
    const auto elementsPerDevice = (elementsCount + deviceCount - 1) / deviceCount;
    TVector<TSlice> slices;
    slices.reserve(deviceCount);

    TSlice slice;
    for (size_t i = 0, inSliceElementCount = 0; i < sizes.size(); ++i) {
        inSliceElementCount += sizes[i];
        if (inSliceElementCount > elementsPerDevice) {
            slice.Right = i + 1;
            slices.push_back(slice);
            slice.Left = slice.Right;
            inSliceElementCount = 0;
        }
    }

    slice.Right = sizes.size();
    slices.push_back(slice);

    // init rest of slices
    slices.insert(slices.begin(), deviceCount - slices.size(), TSlice());
    return TStripeMapping(std::move(slices));
}

static TStripeMapping MakeGroupAwareElementsStripeMappingFromSizes(const TConstArrayRef<ui32> sizes) {
    const auto deviceCount = GetCudaManager().GetDeviceCount();
    const auto elementsCount = Accumulate(sizes.begin(), sizes.end(), size_t(0));
    const auto elementsPerDevice = (elementsCount + deviceCount - 1) / deviceCount;
    TVector<TSlice> slices;
    slices.reserve(deviceCount);

    TSlice slice;
    for (size_t i = 0, inSliceElementCount = 0, totalElementCount = 0; i < sizes.size(); ++i) {
        inSliceElementCount += sizes[i];
        totalElementCount += sizes[i];
        if (inSliceElementCount > elementsPerDevice) {
            slice.Right = totalElementCount;
            slices.push_back(slice);
            slice.Left = slice.Right;
            inSliceElementCount = 0;
        }
    }

    slice.Right = elementsCount;
    slices.push_back(slice);

    // init rest of slices
    slices.insert(slices.begin(), deviceCount - slices.size(), TSlice());
    return TStripeMapping(std::move(slices));
}

static TVector<float> MakeDcgDecays(const TConstArrayRef<ui32> elementwiseOffsets) {
    TVector<float> decays;
    decays.yresize(elementwiseOffsets.size());
    for (ui32 i = 0, iEnd = elementwiseOffsets.size(); i < iEnd; ++i) {
        decays[i] = 1.f / Log2(static_cast<float>(i - elementwiseOffsets[i] + 2));
    }
    return decays;
}

static TVector<float> MakeDcgExponentialDecays(
    const TConstArrayRef<ui32> elementwiseOffsets,
    const float base) {
    TVector<float> decays;
    decays.yresize(elementwiseOffsets.size());
    for (ui32 i = 0, iEnd = elementwiseOffsets.size(); i < iEnd; ++i) {
        decays[i] = pow(base, i - elementwiseOffsets[i]);
    }
    return decays;
}

static TVector<ui64> FuseUi32AndFloatIntoUi64(
    const TConstArrayRef<ui32> ui32s,
    const TConstArrayRef<float> floats) {
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
    const TConstArrayRef<float> floats2) {
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

static void Sort(
    const TConstArrayRef<ui32> ui32s,
    const TConstArrayRef<float> floats,
    const TStripeMapping& mapping,
    TVector<ui32>* const sortedUi32s,
    TVector<float>* const sortedFloats) {
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
    TVector<float>* const sortedFloats2) {
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

                return ui32s[offset + lhs] < ui32s[offset + rhs];
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

static float CalculateIdcg(
    const TConstArrayRef<ui32> sizes,
    const TConstArrayRef<float> weights,
    const TConstArrayRef<float> targets,
    const ENdcgMetricType type,
    const TMaybe<float> exponentialDecay,
    const ui32 topSize) {
    TMaybe<double> doubleDecay;
    if (exponentialDecay.Defined()) {
        doubleDecay = exponentialDecay.GetRef();
    }

    TVector<float> perQueryMetrics;
    perQueryMetrics.yresize(sizes.size());
    TVector<TSample> docs;
    for (size_t i = 0, offset = 0; i < sizes.size(); offset += sizes[i], (void)++i) {
        docs.resize(sizes[i]);
        for (size_t j = 0, jEnd = sizes[i]; j < jEnd; ++j) {
            docs[j].Target = targets[offset + j];
        }

        perQueryMetrics[i] = weights[offset] * CalcIDcg(docs, type, doubleDecay, topSize);
    }

    return FastAccumulate(perQueryMetrics);
}

static float CalculateDcg(
    const TConstArrayRef<ui32> sizes,
    const TConstArrayRef<float> weights,
    const TConstArrayRef<float> targets,
    const TConstArrayRef<float> approxes,
    const ENdcgMetricType type,
    const TMaybe<float> exponentialDecay,
    const ui32 topSize) {
    TMaybe<double> doubleDecay;
    if (exponentialDecay.Defined()) {
        doubleDecay = exponentialDecay.GetRef();
    }

    TVector<float> perQueryMetrics;
    perQueryMetrics.yresize(sizes.size());
    TVector<TSample> docs;
    for (size_t i = 0, offset = 0; i < sizes.size(); offset += sizes[i], (void)++i) {
        docs.resize(sizes[i]);
        for (size_t j = 0, jEnd = sizes[i]; j < jEnd; ++j) {
            docs[j].Target = targets[offset + j];
            docs[j].Prediction = approxes[offset + j];
        }

        perQueryMetrics[i] = weights[offset] * CalcDcg(docs, type, doubleDecay, topSize);
    }

    return FastAccumulate(perQueryMetrics);
}

static float CalculateNdcg(
    const TConstArrayRef<ui32> sizes,
    const TConstArrayRef<float> weights,
    const TConstArrayRef<float> targets,
    const TConstArrayRef<float> approxes,
    const ENdcgMetricType type,
    const ui32 topSize) {
    TVector<float> perQueryMetrics;
    perQueryMetrics.yresize(sizes.size());
    TVector<TSample> docs;
    for (size_t i = 0, offset = 0; i < sizes.size(); offset += sizes[i], (void)++i) {
        docs.resize(sizes[i]);
        for (size_t j = 0, jEnd = sizes[i]; j < jEnd; ++j) {
            docs[j].Target = targets[offset + j];
            docs[j].Prediction = approxes[offset + j];
        }

        TVector<double> decay(sizes[i]);
        FillDcgDecay(ENdcgDenominatorType::LogPosition, Nothing(), decay);
        perQueryMetrics[i] = weights[offset] * CalcNdcg(docs, decay, type, topSize);
    }

    return FastAccumulate(perQueryMetrics);
}

Y_UNIT_TEST_SUITE(NdcgTests) {
    Y_UNIT_TEST(TestMakeDcgDecaySingleDevice) {
        const auto devicesGuard = StartCudaManager();
        const ui32 groupCount = 1000000;
        const ui32 maxGroupSize = 30;
        const ui64 seed = 0;

        const auto offsets = MakeElementwiseOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgDecays(offsets);

        const TSingleMapping mapping(0, offsets.size());
        auto deviceDecay = TSingleBuffer<float>::Create(mapping);
        auto deviceOffsets = TSingleBuffer<ui32>::Create(mapping);

        deviceOffsets.Write(offsets);
        MakeDcgDecays(deviceOffsets, deviceDecay);

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

        const auto offsets = MakeElementwiseOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgDecays(offsets);

        const auto mapping = MakeGroupAwareStripeMappingFromElementwiseOffsets(offsets);
        auto deviceDecay = TStripeBuffer<float>::Create(mapping);
        auto deviceOffsets = TStripeBuffer<ui32>::Create(mapping);

        deviceOffsets.Write(MakeDeviceLocalElementwiseOffsets(offsets, mapping));
        MakeDcgDecays(deviceOffsets, deviceDecay);

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

        const auto offsets = MakeElementwiseOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgExponentialDecays(offsets, base);

        const TSingleMapping mapping(0, offsets.size());
        auto deviceOffsets = TSingleBuffer<ui32>::Create(mapping);
        auto deviceDecay = TSingleBuffer<float>::Create(mapping);

        deviceOffsets.Write(offsets);
        MakeDcgExponentialDecays(deviceOffsets, base, deviceDecay);

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

        const auto offsets = MakeElementwiseOffsets(groupCount, maxGroupSize, seed);
        const auto cpuDecay = ::MakeDcgExponentialDecays(offsets, base);

        const auto mapping = MakeGroupAwareStripeMappingFromElementwiseOffsets(offsets);
        auto deviceDecay = TStripeBuffer<float>::Create(mapping);
        auto deviceOffsets = TStripeBuffer<ui32>::Create(mapping);

        deviceOffsets.Write(MakeDeviceLocalElementwiseOffsets(offsets, mapping));
        MakeDcgExponentialDecays(deviceOffsets, base, deviceDecay);

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
                    // there is some inconsistency between CPU and GPU implementation of float to
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
        const ui32 groupCount = size / 30;
        const ui64 seed = 0;
        const float scaleFactor = 5.f;

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
                    // there is some inconsistency between CPU and GPU implementation of float to
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

        FuseUi32AndTwoFloatsIntoUi64(deviceUi32s, deviceFloats1, deviceFloats2, deviceFused, true, false);
        MakeSequence(deviceIndices);
        RadixSort(deviceFused, deviceIndices);
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

    static void TestIdcg(
        const size_t size,
        const size_t maxDocsPerQuery,
        const ui64 seed,
        const ENdcgMetricType type,
        const TMaybe<float> exponentialDecay,
        const bool withWeights,
        const ui32 topSize,
        const float eps) {
        const auto deviceGuard = StartCudaManager();
        const float scale = 5;
        const float weightsScale = 1;

        TFastRng<ui64> prng(seed);
        TVector<ui32> sizes;
        sizes.reserve(size);
        TVector<ui32> offsets;
        offsets.reserve(size);
        for (size_t curSize = 0; curSize < size; curSize += sizes.back()) {
            sizes.push_back(Min<ui32>(prng.Uniform(maxDocsPerQuery) + 1, size - curSize));
            offsets.push_back(curSize);
        }
        TVector<float> targets;
        targets.yresize(size);
        for (auto& target : targets) {
            target = prng.GenRandReal1() * scale;
        }
        TVector<float> weights;
        weights.yresize(size);
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto weight = withWeights ? prng.GenRandReal1() * weightsScale : 1.f;
            for (ui32 j = 0; j < sizes[i]; ++j) {
                weights[offsets[i] + j] = weight;
            }
        }

        const auto sizesMapping = MakeGroupAwareStripeMappingFromSizes(sizes);
        const auto elementsMapping = MakeGroupAwareElementsStripeMappingFromSizes(sizes);
        auto deviceSizes = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceBiasedOffsets = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceOffsetsBias = MakeOffsetsBias(offsets, sizesMapping);
        auto deviceWeights = TStripeBuffer<float>::Create(elementsMapping);
        auto deviceTargets = TStripeBuffer<float>::Create(elementsMapping);

        deviceSizes.Write(sizes);
        deviceBiasedOffsets.Write(offsets);
        deviceWeights.Write(weights);
        deviceTargets.Write(targets);

        const auto cpuIdcg = CalculateIdcg(
                                 sizes,
                                 weights,
                                 targets,
                                 type,
                                 exponentialDecay,
                                 topSize) /
                             sizes.size();
        const auto gpuIdcg = CalculateIdcg(
                                 deviceSizes.ConstCopyView(),
                                 deviceBiasedOffsets.ConstCopyView(),
                                 deviceOffsetsBias,
                                 deviceWeights.ConstCopyView(),
                                 deviceTargets.ConstCopyView(),
                                 type,
                                 exponentialDecay,
                                 {topSize})
                                 .front() /
                             sizes.size();
        UNIT_ASSERT_DOUBLES_EQUAL_C(
            cpuIdcg, gpuIdcg, eps,
            LabeledOutput(size, maxDocsPerQuery, seed, type, exponentialDecay));
    }

    Y_UNIT_TEST(TestIdcg) {
        const ui64 seed = 0;
        const TMaybe<float> exponentialDecays[] = {Nothing(), 0.5};
        const ui32 topSizes[] = {5, 10, Max<ui32>()};
        for (const size_t size : {1000, 10000, 10000000}) {
            for (const size_t maxDocsPerQuery : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 40}) {
                for (const auto type : {ENdcgMetricType::Base, ENdcgMetricType::Exp}) {
                    for (const auto exponentialDecay : exponentialDecays) {
                        for (const auto withWeights : {false, true}) {
                            for (const auto topSize : topSizes) {
                                float eps = 1e-5;
                                if (size == 10000000 || maxDocsPerQuery == 40) {
                                    eps = 1e-4;
                                }
                                TestIdcg(size, maxDocsPerQuery, seed, type, exponentialDecay, withWeights, topSize, eps);
                            }
                        }
                    }
                }
            }
        }
    }

    static void TestDcg(
        const size_t size,
        const size_t maxDocsPerQuery,
        const ui64 seed,
        const ENdcgMetricType type,
        const TMaybe<float> exponentialDecay,
        const bool withWeights,
        const ui32 topSize,
        const float eps) {
        const auto deviceGuard = StartCudaManager();
        const float scale = 5;
        const float weightsScale = 1;

        TFastRng<ui64> prng(seed);
        TVector<ui32> sizes;
        sizes.reserve(size);
        TVector<ui32> offsets;
        offsets.reserve(size);
        for (size_t curSize = 0; curSize < size; curSize += sizes.back()) {
            sizes.push_back(Min<ui32>(prng.Uniform(maxDocsPerQuery) + 1, size - curSize));
            offsets.push_back(curSize);
        }

        // `CalculateDcg` on GPU operates float16 when does sorting, if we leave bits in mantissa
        // that get lost when rounding from float to float16 we may get a different sorting order
        // which in turn may result in significant difference in metric value (up to ones on 10^6
        // randomly generated documents with relevance in [0; 5]), its manifestation especially
        // visible on Exp variant of metric where numerator of summand is 2^relevance_i.
        TVector<float> targets;
        targets.yresize(size);
        for (auto& target : targets) {
            target = TFloat16(prng.GenRandReal1() * scale);
        }
        TVector<float> approxes;
        approxes.yresize(size);
        for (auto& approx : approxes) {
            approx = TFloat16(prng.GenRandReal1() * scale);
        }
        TVector<float> weights;
        weights.yresize(size);
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto weight = withWeights ? prng.GenRandReal1() * weightsScale : 1.f;
            for (ui32 j = 0; j < sizes[i]; ++j) {
                weights[offsets[i] + j] = weight;
            }
        }

        const auto sizesMapping = MakeGroupAwareStripeMappingFromSizes(sizes);
        const auto elementsMapping = MakeGroupAwareElementsStripeMappingFromSizes(sizes);
        auto deviceSizes = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceBiasedOffsets = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceOffsetsBias = MakeOffsetsBias(offsets, sizesMapping);
        auto deviceWeights = TStripeBuffer<float>::Create(elementsMapping);
        auto deviceTargets = TStripeBuffer<float>::Create(elementsMapping);
        auto deviceApproxes = TStripeBuffer<float>::Create(elementsMapping);

        deviceSizes.Write(sizes);
        deviceBiasedOffsets.Write(offsets);
        deviceWeights.Write(weights);
        deviceTargets.Write(targets);
        deviceApproxes.Write(approxes);

        const auto cpuDcg = CalculateDcg(
                                sizes,
                                weights,
                                targets,
                                approxes,
                                type,
                                exponentialDecay,
                                topSize) /
                            sizes.size();
        const auto gpuDcg = CalculateDcg(
                                deviceSizes.ConstCopyView(),
                                deviceBiasedOffsets.ConstCopyView(),
                                deviceOffsetsBias,
                                deviceWeights.ConstCopyView(),
                                deviceTargets.ConstCopyView(),
                                deviceApproxes.ConstCopyView(),
                                type,
                                exponentialDecay,
                                {topSize})
                                .front() /
                            sizes.size();
        UNIT_ASSERT_DOUBLES_EQUAL_C(
            cpuDcg, gpuDcg, eps,
            LabeledOutput(size, maxDocsPerQuery, seed, type, exponentialDecay));
    }

    Y_UNIT_TEST(TestDcg) {
        const ui64 seed = 0;
        const TMaybe<float> exponentialDecays[] = {Nothing(), 0.5};
        const ui32 topSizes[] = {5, 10, Max<ui32>()};
        for (const size_t size : {1000, 10000, 10000000}) {
            for (const size_t maxDocsPerQuery : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 40}) {
                for (const auto type : {ENdcgMetricType::Base, ENdcgMetricType::Exp}) {
                    for (const auto exponentialDecay : exponentialDecays) {
                        for (const auto withWeights : {false, true}) {
                            for (const auto topSize : topSizes) {
                                float eps = 1e-5;
                                if (size == 10000000) {
                                    eps = 1e-4;
                                }
                                if (size == 10000000 && type == ENdcgMetricType::Exp) {
                                    eps = 1e-3;
                                }
                                if (size == 10000 && maxDocsPerQuery == 31) {
                                    eps = 1e-3;
                                }
                                TestDcg(size, maxDocsPerQuery, seed, type, exponentialDecay, withWeights, topSize, eps);
                            }
                        }
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestMakeEndOfGroupMarkers) {
        const auto deviceGuard = StartCudaManager();
        const size_t size = 10000000;
        const size_t maxGroupSize = 30;
        const ui64 seed = 0;

        TFastRng<ui64> prng(seed);
        TVector<ui32> sizes;
        sizes.reserve(size / maxGroupSize);
        TVector<ui32> offsets;
        sizes.reserve(sizes.size());
        for (size_t curSize = 0; curSize < size;) {
            sizes.push_back(Min<ui32>(prng.Uniform(maxGroupSize) + 1, size - curSize));
            offsets.push_back(curSize);
            curSize += sizes.back();
        }

        TVector<ui32> cpuEndOfGroupMarkers;
        cpuEndOfGroupMarkers.resize(size);
        cpuEndOfGroupMarkers[0] = 1;
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (const auto offset = offsets[i] + sizes[i]; offset < size) {
                cpuEndOfGroupMarkers[offset] = 1;
            }
        }

        const TSingleMapping sizesMapping(0, sizes.size());
        auto deviceSizes = TSingleBuffer<ui32>::Create(sizesMapping);
        auto deviceBiasedOffsets = TSingleBuffer<ui32>::Create(sizesMapping);
        auto deviceOffsetsBias = GetCudaManager().CreateDistributedObject<ui32>(0);
        const TSingleMapping endOfGroupMarkersMapping(0, size);
        auto deviceEndOfGroupMarkers = TSingleBuffer<ui32>::Create(endOfGroupMarkersMapping);

        deviceSizes.Write(sizes);
        deviceBiasedOffsets.Write(offsets);

        FillBuffer(deviceEndOfGroupMarkers, ui32(0));
        MakeEndOfGroupMarkers(deviceSizes, deviceBiasedOffsets, deviceOffsetsBias, deviceEndOfGroupMarkers);

        TVector<ui32> gpuEndOfGroupMarkers;
        deviceEndOfGroupMarkers.Read(gpuEndOfGroupMarkers);

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(
                cpuEndOfGroupMarkers[i], gpuEndOfGroupMarkers[i],
                LabeledOutput(i, size, maxGroupSize));
        }
    }

    void TestMakeElementwiseOffsets(const size_t size, const size_t maxGroupSize, const ui64 seed) {
        const auto deviceGuard = StartCudaManager();

        TFastRng<ui64> prng(seed);
        TVector<ui32> sizes;
        sizes.reserve(size);
        TVector<ui32> offsets;
        sizes.reserve(sizes.size());
        for (size_t curSize = 0; curSize < size; curSize += sizes.back()) {
            sizes.push_back(Min<ui32>(prng.Uniform(maxGroupSize) + 1, size - curSize));
            offsets.push_back(curSize);
        }

        const auto sizesMapping = MakeGroupAwareStripeMappingFromSizes(sizes);
        const auto elementsMapping = MakeGroupAwareElementsStripeMappingFromSizes(sizes);
        auto deviceSizes = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceBiasedOffsets = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceOffsetsBias = MakeOffsetsBias(offsets, sizesMapping);
        auto deviceElementwiseOffsets = TStripeBuffer<ui32>::Create(elementsMapping);

        deviceSizes.Write(sizes);
        deviceBiasedOffsets.Write(offsets);

        MakeElementwiseOffsets(deviceSizes, deviceBiasedOffsets, deviceOffsetsBias, deviceElementwiseOffsets);

        TVector<ui32> gpuElementwiseOffsets;
        deviceElementwiseOffsets.Read(gpuElementwiseOffsets);

        TVector<ui32> cpuElementwiseOffsets;
        cpuElementwiseOffsets.yresize(size);
        for (size_t i = 0; i < sizes.size(); ++i) {
            for (size_t j = offsets[i], jEnd = offsets[i] + sizes[i]; j < jEnd; ++j) {
                cpuElementwiseOffsets[j] = offsets[i];
            }
        }
        cpuElementwiseOffsets = MakeDeviceLocalElementwiseOffsets(cpuElementwiseOffsets, elementsMapping);

        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT_VALUES_EQUAL_C(
                cpuElementwiseOffsets[i], gpuElementwiseOffsets[i],
                LabeledOutput(i, size, maxGroupSize));
        }
    }

    Y_UNIT_TEST(TestMakeElementwiseOffsets) {
        const ui64 seed = 0;
        for (const size_t size : {100, 999, 998, 997, 996, 995, 994, 1000000}) {
            for (const size_t maxGroupSize : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 40}) {
                TestMakeElementwiseOffsets(size, maxGroupSize, seed);
            }
        }
    }

    static void TestNdcg(
        const size_t size,
        const size_t maxDocsPerQuery,
        const ui64 seed,
        const ENdcgMetricType type,
        const bool withWeights,
        const ui32 topSize,
        const float eps) {
        const auto deviceGuard = StartCudaManager();
        const float scale = 5;
        const float weightsScale = 1;

        TFastRng<ui64> prng(seed);
        TVector<ui32> sizes;
        sizes.reserve(size);
        TVector<ui32> offsets;
        offsets.reserve(size);
        for (size_t curSize = 0; curSize < size; curSize += sizes.back()) {
            sizes.push_back(Min<ui32>(prng.Uniform(maxDocsPerQuery) + 1, size - curSize));
            offsets.push_back(curSize);
        }
        TVector<float> targets;
        targets.yresize(size);
        for (auto& target : targets) {
            target = TFloat16(prng.GenRandReal1() * scale);
        }
        TVector<float> approxes;
        approxes.yresize(size);
        for (auto& approx : approxes) {
            approx = TFloat16(prng.GenRandReal1() * scale);
        }
        TVector<float> weights;
        weights.yresize(size);
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto weight = withWeights ? prng.GenRandReal1() * weightsScale : 1.f;
            for (ui32 j = 0; j < sizes[i]; ++j) {
                weights[offsets[i] + j] = weight;
            }
        }

        const auto sizesMapping = MakeGroupAwareStripeMappingFromSizes(sizes);
        const auto elementsMapping = MakeGroupAwareElementsStripeMappingFromSizes(sizes);
        auto deviceSizes = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceBiasedOffsets = TStripeBuffer<ui32>::Create(sizesMapping);
        auto deviceOffsetsBias = MakeOffsetsBias(offsets, sizesMapping);
        auto deviceWeights = TStripeBuffer<float>::Create(elementsMapping);
        auto deviceTargets = TStripeBuffer<float>::Create(elementsMapping);
        auto deviceApproxes = TStripeBuffer<float>::Create(elementsMapping);

        deviceSizes.Write(sizes);
        deviceBiasedOffsets.Write(offsets);
        deviceWeights.Write(weights);
        deviceTargets.Write(targets);
        deviceApproxes.Write(approxes);

        const auto cpuNdcg = CalculateNdcg(
                                 sizes,
                                 weights,
                                 targets,
                                 approxes,
                                 type,
                                 topSize) /
                             sizes.size();
        const auto gpuNdcg = CalculateNdcg(
                                 deviceSizes.ConstCopyView(),
                                 deviceBiasedOffsets.ConstCopyView(),
                                 deviceOffsetsBias,
                                 deviceWeights.ConstCopyView(),
                                 deviceTargets.ConstCopyView(),
                                 deviceApproxes.ConstCopyView(),
                                 type,
                                 {topSize})
                                 .front() /
                             sizes.size();
        UNIT_ASSERT_DOUBLES_EQUAL_C(
            cpuNdcg, gpuNdcg, eps,
            LabeledOutput(size, maxDocsPerQuery, seed, type));
    }

    Y_UNIT_TEST(TestNdcg) {
        const ui64 seed = 0;
        const ui32 topSizes[] = {5, 10, Max<ui32>()};
        for (const size_t size : {1000, 10000, 10000000}) {
            for (const size_t maxDocsPerQuery : {1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 40}) {
                for (const auto type : {ENdcgMetricType::Base, ENdcgMetricType::Exp}) {
                    for (const auto withWeights : {false, true}) {
                        for (const auto topSize : topSizes) {
                            TestNdcg(size, maxDocsPerQuery, seed, type, withWeights, topSize, 1e-5);
                        }
                    }
                }
            }
        }
    }
}
