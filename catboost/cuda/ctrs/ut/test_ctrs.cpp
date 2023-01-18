#include "calc_ctr_cpu.h"
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/ctrs/ctr_bins_builder.h>
#include <catboost/cuda/ctrs/ctr_calcers.h>
#include <catboost/cuda/data/data_utils.h>
#include <util/generic/hash.h>
#include <util/random/shuffle.h>
#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/threading/local_executor/local_executor.h>
#include <iostream>

using namespace std;
using namespace NCudaLib;
using namespace NCatboostCuda;

Y_UNIT_TEST_SUITE(TCtrTest) {
    template <class T>
    TVector<T> BuildRandomBins(TRandom & rand, ui32 uniqueValues, ui32 sampleCount) {
        TVector<T> bins(sampleCount);
        for (ui32 i = 0; i < sampleCount; ++i) {
            bins[i] = rand.NextUniformL() % uniqueValues;
        }
        return bins;
    }

    void TestSimpleCtr(bool useGroups) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            NPar::LocalExecutor().RunAdditionalThreads(8);
            ui64 tries = 5;
            ui64 meanGroupSize = useGroups ? 7 : 1;
            const ui32 uniqueValues = 15542;

            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            TRandom rand(0);
            for (ui32 run = 0; run < tries; ++run) {
                const ui64 groupCount = (500000 + rand.NextUniformL() % 1000) / meanGroupSize;
                TVector<ui64> groupIds;

                ui64 learnSize = 0;
                if (useGroups) {
                    for (ui64 group = 0; group < groupCount; ++group) {
                        ui64 groupSize = 1 + (rand.NextUniformL() % meanGroupSize);
                        for (ui64 i = 0 ; i < groupSize; ++i) {
                            groupIds.push_back(group);
                        }

                        if (group ==  (groupCount - 300)) {
                            learnSize = groupIds.size();
                        }
                    }
                } else {
                    groupIds.resize(groupCount);
                    Iota(groupIds.begin(), groupIds.end(), 0);
                    learnSize = groupCount - 10000;
                }

                const ui64 size = groupIds.size();

                auto weights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(size));
                auto learnSlice = TSlice(0, learnSize);

                TVector<float> cpuWeights;

                for (ui32 i = 0; i < size; ++i) {
                    auto weight = i < learnSize ? 1.0f / (1 << (rand.NextUniformL() % 4)) : 0.0f;
                    //                    auto weight = i < learnSize ? 1.0f : 0.0f;
                    cpuWeights.push_back(weight);
                }
                weights.Write(cpuWeights);

                auto indices = TMirrorBuffer<ui32>::CopyMapping(weights);
                auto groupIndices = [&]() {
                    TVector<ui32> groupIdsForGpu;
                    for (auto val : groupIds) {
                        groupIdsForGpu.push_back(val);
                    }
                    auto tmp = TMirrorBuffer<ui32>::CopyMapping(weights);
                    tmp.Write(groupIdsForGpu);
                    return tmp.ConstCopyView();
                }();


                TVector<ui32> cpuIndices;
                {
                    cpuIndices.resize(size);
                    std::iota(cpuIndices.begin(), cpuIndices.end(), 0);
                    indices.Write(cpuIndices);

//
                    TVector<ui32> shuffled;
                    TConstArrayRef<TGroupId> groupIdsArrayRef = groupIds;
                    QueryConsistentShuffle(rand.NextUniformL(), 1, groupIdsArrayRef.Slice(0, learnSize), &shuffled);
                    indices.SliceView(learnSlice).Write(shuffled);

                    for (ui64 i = 0; i < shuffled.size(); ++i) {
                        cpuIndices[i] = shuffled[i];
                    }

                }

                auto bins = BuildRandomBins<ui32>(rand, uniqueValues, size);

                auto binIndices = [&]() -> TMirrorBuffer<ui32> {
                    TCtrBinBuilder<TMirrorMapping> ctrBinBuilder;
                    ctrBinBuilder
                        .SetIndices(indices);

                    auto binsGpu = TMirrorBuffer<ui32>::CopyMapping(weights);
                    binsGpu.Write(bins);

                    auto compressedBinsGpu = TMirrorBuffer<ui64>::Create(CompressedSize<ui64>(binsGpu, uniqueValues));
                    Compress(binsGpu, compressedBinsGpu, uniqueValues);

                    return ctrBinBuilder
                        .AddCompressedBins(compressedBinsGpu, uniqueValues)
                        .MoveIndices();
                }();

//                auto groupIds
                THistoryBasedCtrCalcer<NCudaLib::TMirrorMapping> builder(weights, binIndices, useGroups ? &groupIndices : nullptr);

                for (ui32 numClasses : {2, 3, 4, 8, 16, 32, 55, 255}) {
                    auto targets = BuildRandomBins<ui8>(rand, numClasses, size);
                    auto targetsGpu = TMirrorBuffer<ui8>::CopyMapping(indices);
                    targetsGpu.Write(targets);

                    const float prior = 0.5;
                    const float priorDenum = 1.0;
                    TCpuTargetClassCtrCalcer ctrCalcer(uniqueValues, bins, cpuWeights, prior, priorDenum);
                    auto ctrs = ctrCalcer.Calc(cpuIndices, TConstArrayRef<ui64>(groupIds), targets, numClasses);

                    TMirrorBuffer<float> cudaCtr;

                    TVector<float> priorParams = {0.5, 1};
                    builder.SetBinarizedSample(targetsGpu.ConstCopyView());

                    for (ui32 clazz = 0; clazz < numClasses; ++clazz) {
                        NCB::TCtrConfig config;
                        config.Prior = priorParams;
                        config.ParamId = clazz;
                        config.Type = ECtrType::Buckets;
                        {
                            auto guard = profiler.Profile(
                                TStringBuilder() << "Compute target ctr for unique values #" << uniqueValues
                                                 << " and classes #" << numClasses);
                            builder.ComputeCatFeatureCtr(config, cudaCtr);
                        }

                        TVector<float> ctrsFromGpu;
                        cudaCtr.Read(ctrsFromGpu);

                        for (ui32 i = 0; i < cpuIndices.size(); ++i) {
                            UNIT_ASSERT_DOUBLES_EQUAL(ctrs[i][clazz], ctrsFromGpu[cpuIndices[i]], 1e-6f);
                        }
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestSimpleCatTargetCtr) {
        TestSimpleCtr(false);
    }

    Y_UNIT_TEST(TestGroupwiseSimpleCatTargetCtr) {
        TestSimpleCtr(true);
    }

    Y_UNIT_TEST(TestSimpleCatTargetCtrBenchmark) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            NPar::LocalExecutor().RunAdditionalThreads(8);
            ui64 tries = 20;
            const ui32 uniqueValues = 155542;

            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            TRandom rand(0);
            for (ui32 run = 0; run < tries; ++run) {
                const ui64 size = 10000000 + rand.NextUniformL() % 1000;

                auto weights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(size));
                const auto learnSize = size - 10000;
                auto learnSlice = TSlice(0, learnSize);

                TVector<float> cpuWeights;

                for (ui32 i = 0; i < size; ++i) {
                    auto weight = i < learnSize ? 1.0f / (1 << (rand.NextUniformL() % 4)) : 0.0f;
                    //                    auto weight = i < learnSize ? 1.0f : 0.0f;
                    cpuWeights.push_back(weight);
                }
                weights.Write(cpuWeights);

                auto indices = TMirrorBuffer<ui32>::CopyMapping(weights);
                TVector<ui32> cpuIndices;

                {
                    cpuIndices.resize(size);
                    std::iota(cpuIndices.begin(), cpuIndices.end(), 0);
                    indices.Write(cpuIndices);

                    Shuffle(cpuIndices.begin(), cpuIndices.begin() + learnSize, rand);
                    indices.SliceView(learnSlice).Write(cpuIndices);
                }

                auto bins = BuildRandomBins<ui32>(rand, uniqueValues, size);

                auto binIndices = [&]() -> TMirrorBuffer<ui32> {
                    TCtrBinBuilder<TMirrorMapping> ctrBinBuilder;
                    ctrBinBuilder
                        .SetIndices(indices);

                    auto binsGpu = TMirrorBuffer<ui32>::CopyMapping(weights);
                    binsGpu.Write(bins);

                    auto compressedBinsGpu = TMirrorBuffer<ui64>::Create(CompressedSize<ui64>(binsGpu, uniqueValues));
                    Compress(binsGpu, compressedBinsGpu, uniqueValues);

                    return ctrBinBuilder
                        .AddCompressedBins(compressedBinsGpu, uniqueValues)
                        .MoveIndices();
                }();

                THistoryBasedCtrCalcer<NCudaLib::TMirrorMapping> builder(weights, binIndices, nullptr);

                for (ui32 numClasses : {2, 3, 4, 8, 16, 32, 55, 255}) {
                    auto targets = BuildRandomBins<ui8>(rand, numClasses, size);
                    auto targetsGpu = TMirrorBuffer<ui8>::CopyMapping(indices);
                    targetsGpu.Write(targets);
                    builder.SetBinarizedSample(targetsGpu.AsConstBuf());

                    const float prior = 0.5;

                    TMirrorBuffer<float> cudaCtr;

                    TVector<float> priorParams(numClasses, prior);
                    for (ui32 clazz = 0; clazz < numClasses - 1; ++clazz) {
                        NCB::TCtrConfig config;
                        config.Prior = priorParams;
                        config.ParamId = clazz;
                        config.Type = ECtrType::Buckets;
                        {
                            auto guard = profiler.Profile(
                                TStringBuilder() << "Compute target ctr for unique values #" << uniqueValues
                                                 << " and classes #" << numClasses);
                            builder.ComputeCatFeatureCtr(config, cudaCtr);
                        }
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestSimpleFreqCtr) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            NPar::LocalExecutor().RunAdditionalThreads(8);
            ui64 maxBits = 25;
            ui64 tries = 5;

            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();
            SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

            TRandom rand(0);
            for (ui32 run = 0; run < tries; ++run) {
                const ui64 size = 1000000 + rand.NextUniformL() % 10000;

                auto weights = TMirrorBuffer<float>::Create(NCudaLib::TMirrorMapping(size));
                const auto learnSize = size - 10000;
                auto learnSlice = TSlice(0, learnSize);

                TVector<float> cpuWeights;
                TVector<float> cpuWeights2;
                float totalWeight = 0;
                for (ui32 i = 0; i < size; ++i) {
                    auto weight = i < learnSize ? 1.0f / (1 << (rand.NextUniformL() % 4)) : 0.0f;
                    cpuWeights.push_back(weight);
                    cpuWeights2.push_back(1.0f);
                    totalWeight += cpuWeights.back();
                }
                weights.Write(cpuWeights);

                auto indices = TMirrorBuffer<ui32>::CopyMapping(weights);
                TVector<ui32> cpuIndices;

                {
                    cpuIndices.resize(size);
                    std::iota(cpuIndices.begin(), cpuIndices.end(), 0);
                    indices.Write(cpuIndices);

                    Shuffle(cpuIndices.begin(), cpuIndices.begin() + learnSize, rand);
                    indices.SliceView(learnSlice).Write(cpuIndices);
                }

                for (ui32 bits = 1; bits < maxBits; ++bits) {
                    ui32 uniqueValues = (ui32)(1 << bits);
                    auto bins = BuildRandomBins<ui32>(rand, uniqueValues, size);

                    auto binsGpu = TMirrorBuffer<ui32>::CopyMapping(weights);
                    binsGpu.Write(bins);

                    auto compressedBinsGpu = TMirrorBuffer<ui64>::Create(CompressedSize<ui64>(binsGpu,
                                                                                              uniqueValues));
                    FillBuffer(compressedBinsGpu, static_cast<ui64>(0));
                    Compress(binsGpu, compressedBinsGpu, uniqueValues);

                    TWeightedBinFreqCalcer<NCudaLib::TMirrorMapping> builder(weights,
                                                                             totalWeight);

                    TMirrorBuffer<float> ctrs;
                    TMirrorBuffer<float> ctrs2;

                    NCB::TCtrConfig config = CreateCtrConfigForFeatureFreq(0.5, uniqueValues);

                    TCpuTargetClassCtrCalcer calcer(uniqueValues, bins, cpuWeights, GetNumeratorShift(config), GetDenumeratorShift(config));
                    TCpuTargetClassCtrCalcer calcer2(uniqueValues, bins, cpuWeights2, GetNumeratorShift(config), GetDenumeratorShift(config));
                    auto ctrsCpu = calcer.ComputeFreqCtr();
                    auto ctrsCpu2 = calcer2.ComputeFreqCtr();

                    {
                        TCtrBinBuilder<TMirrorMapping> ctrBinBuilder;
                        ctrBinBuilder
                            .SetIndices(indices)
                            .AddCompressedBins(compressedBinsGpu,
                                               uniqueValues);

                        auto guard = profiler.Profile(TStringBuilder() << "Compute freq for unique values #" << uniqueValues);
                        builder.ComputeFreq(ctrBinBuilder.GetIndices().ConstCopyView(),
                                            config,
                                            ctrs);

                        decltype(ctrBinBuilder)::TVisitor ctrVisitor = [&](const NCB::TCtrConfig& ctrConfig,
                                                                           const TCudaBuffer<float, TMirrorMapping>& ctrValues,
                                                                           ui32 stream) {
                            Y_UNUSED(ctrConfig);
                            Y_UNUSED(stream);
                            ctrs2 = TMirrorBuffer<float>::CopyMapping(ctrValues);
                            ctrs2.Copy(ctrValues);
                        };
                        ctrBinBuilder.VisitEqualUpToPriorFreqCtrs(SingletonVector(config),
                                                                  ctrVisitor);
                    }

                    TVector<float> ctrsFromGpu;
                    TVector<float> ctrs2FromGpu;
                    ctrs.Read(ctrsFromGpu);
                    ctrs2.Read(ctrs2FromGpu);

                    NPar::ParallelFor(0, cpuIndices.size(), [&](int i) {
                        const ui32 idx = cpuIndices[i];
                        UNIT_ASSERT_DOUBLES_EQUAL(ctrsCpu[idx], ctrsFromGpu[idx], 1e-6f);
                        UNIT_ASSERT_DOUBLES_EQUAL(ctrsCpu2[idx], ctrs2FromGpu[idx], 1e-6f);
                    });
                }
            }
        }
    }
}
