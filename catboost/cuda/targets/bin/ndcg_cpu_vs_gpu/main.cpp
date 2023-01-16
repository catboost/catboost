#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/targets/dcg.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/getopt/last_getopt.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/algorithm.h>
#include <util/generic/map.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/random/fast.h>
#include <util/string/join.h>
#include <util/system/hp_timer.h>
#include <util/system/info.h>
#include <util/system/types.h>

using NCatboostCuda::CalculateNdcg;
using NCatboostCuda::NDetail::GatherBySizeAndOffset;
using NCudaLib::GetCudaManager;
using NCudaLib::TDeviceRequestConfig;
using NCudaLib::TDistributedObject;
using NCudaLib::TStripeMapping;

static ui32 GetDefaultCpuThreadCount() {
    const auto cpuCount = NSystemInfo::CachedNumberOfCpus();
    return (cpuCount + 1) / 2;
}

namespace {
    struct TArgs {
        ui32 MaxDocumentsPerQuery = 0;
        ui32 DocumentCount = 0;
        float RelevanceScale = 5.f;
        float ApproxMultiplier = 1.f;
        ui64 RandomGeneratorSeed = 20181203;
        ui32 IterationCount = 5;

        ui32 CpuThreadCount = GetDefaultCpuThreadCount();
        // TString GpuDeviceIds = "-1";

        static TArgs ParseArgs(const int argc, const char* argv[]);
    };
}

TArgs TArgs::ParseArgs(const int argc, const char* argv[]) {
    TArgs args;
    auto p = NLastGetopt::TOpts::Default();
    p.AddLongOption("max-docs-per-query")
     .Required()
     .RequiredArgument("INT")
     .StoreResult(&args.MaxDocumentsPerQuery);
    p.AddLongOption("document-count")
     .Required()
     .RequiredArgument("INT")
     .StoreResult(&args.DocumentCount);
    p.AddLongOption("relevance-scale")
     .DefaultValue(ToString(args.RelevanceScale))
     .RequiredArgument("FLOAT")
     .StoreResult(&args.RelevanceScale);
    p.AddLongOption("approx-multiplier")
     .DefaultValue(ToString(args.ApproxMultiplier))
     .RequiredArgument("FLOAT")
     .StoreResult(&args.ApproxMultiplier);
    p.AddLongOption("random-seed")
     .DefaultValue(ToString(args.RandomGeneratorSeed))
     .RequiredArgument("INT")
     .StoreResult(&args.RandomGeneratorSeed);
    p.AddLongOption('j', "cpu-thread-count")
     .DefaultValue(ToString(args.CpuThreadCount))
     .RequiredArgument("INT")
     .StoreResult(&args.CpuThreadCount);
    /*p.AddLongOption("gpu-device-ids")
     .DefaultValue(args.GpuDeviceIds)
     .RequiredArgument("LIST")
     .StoreResult(&args.GpuDeviceIds);
     */
    p.AddLongOption("iteration-count")
     .DefaultValue(ToString(args.IterationCount))
     .RequiredArgument("INT")
     .StoreResult(&args.IterationCount);
    p.SetFreeArgsNum(0);
    NLastGetopt::TOptsParseResult(&p, argc, argv);

    return args;
}

namespace {
    struct TData {
        ui32 QueryCount = 0;
        ui32 DocumentCount = 0;
        TVector<ui32> Sizes;
        TVector<float> Weights;
        TVector<float> Targets;
        TVector<float> Approxes;

        static TData Generate(
            ui64 randomSeed,
            ui32 maxDocumentsPerQuery,
            ui32 documentCount,
            float relevanceScale,
            float approxMultiplier);
    };
}

TData TData::Generate(
    const ui64 randomSeed,
    const ui32 maxDocumentsPerQuery,
    const ui32 documentCount,
    const float relevanceScale,
    const float approxMultiplier)
{
    TData data;

    data.DocumentCount = documentCount;
    data.Sizes.reserve((documentCount + maxDocumentsPerQuery - 1) / maxDocumentsPerQuery);

    TFastRng<ui64> prng(randomSeed);
    for (ui32 curDocCnt = 0; curDocCnt < documentCount; curDocCnt += data.Sizes.back()) {
        data.Sizes.push_back(Min<ui32>(prng.Uniform(maxDocumentsPerQuery) + 1, documentCount - curDocCnt));
    }

    data.QueryCount = data.Sizes.size();
    data.Weights.yresize(data.QueryCount);
    data.Targets.yresize(documentCount);
    data.Approxes.yresize(documentCount);

    for (auto& weight : data.Weights) {
        weight = prng.GenRandReal1();
    }

    for (auto& target : data.Targets) {
        target = prng.GenRandReal1() * relevanceScale;
    }

    for (auto& approx : data.Approxes) {
        approx = prng.GenRandReal1() * relevanceScale * approxMultiplier;
    }

    return data;
}

namespace {
    struct TMetricParams {
        ui32 Top = 0;
        ENdcgMetricType Type = ENdcgMetricType::Base;
        bool UseWeights = false;

        static TString MakeDescription(const TMetricParams& params);
    };
}

TString TMetricParams::MakeDescription(const TMetricParams& params) {
    return TString::Join("NDCG:",
        "top=", ToString(params.Top),
        ";type=", ToString(params.Type),
        ";use_weights=", params.UseWeights ? "true" : "false");
}

static double RunOnCpu(
    const TData& data,
    const ui32 threadCount,
    const TMetricParams& params,
    const ui32 iterationCount)
{
    THPTimer timer;

    CATBOOST_INFO_LOG << "initializing data for CPU evaluator" << '\n';
    TVector<TQueryInfo> queryInfos;
    queryInfos.reserve(data.QueryCount);
    for (ui32 i = 0, iEnd = data.QueryCount, offset = 0; i < iEnd; offset += data.Sizes[i], (void)++i) {
        TQueryInfo queryInfo(offset, offset + data.Sizes[i]);
        queryInfo.Weight = data.Weights[i];
        queryInfos.emplace_back(std::move(queryInfo));
    }

    TVector<TVector<double>> approxes(1);
    approxes.front().assign(data.Approxes.begin(), data.Approxes.end());
    CATBOOST_INFO_LOG << "initialized data for CPU evaluator in " << timer.Passed() << " sec." << 'n';

    const auto description = TMetricParams::MakeDescription(params);
    const auto evaluator = std::move(CreateMetricsFromDescription({description}, 1).front());
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);

    TVector<float> dummyWeights;

    double bestTime = Max<double>();
    for (ui32 iteration = 0; iteration < iterationCount; ++iteration) {
        CATBOOST_INFO_LOG << "cpu: " << LabeledOutput(description, threadCount, iteration) << "; started" << '\n';
        timer.Reset();
        const auto ndcg = dynamic_cast<const ISingleTargetEval*>(evaluator.Get())->Eval(
            approxes,
            data.Targets,
            dummyWeights,
            queryInfos,
            0,
            data.QueryCount,
            executor);
        Y_FAKE_READ(ndcg);
        const auto iterationTime = timer.Passed();
        bestTime = Min(bestTime, iterationTime);
        CATBOOST_INFO_LOG << "cpu: " << LabeledOutput(description, threadCount, iteration, iterationTime, bestTime) << '\n';
    }

    CATBOOST_INFO_LOG << "cpu: " << LabeledOutput(description, bestTime) << '\n';

    return bestTime;
}

static TStripeMapping MakeDeviceAwareGroupsMapping(
    const TConstArrayRef<ui32> sizes,
    const ui32 sizesSum)
{
    const ui32 deviceCount = GetCudaManager().GetDeviceCount();
    const ui32 elementsPerDevice = (sizesSum + deviceCount - 1) / deviceCount;
    TVector<TSlice> slices;
    slices.reserve(deviceCount);
    TSlice slice;
    for (ui32 i = 0, iEnd = sizes.size(), inSliceElementCount = 0; i < iEnd; ++i) {
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

    return TStripeMapping(std::move(slices));
}

static TStripeMapping MakeDeviceAwareElementsMapping(
    const TConstArrayRef<ui32> sizes,
    const ui32 sizesSum)
{
    const ui32 deviceCount = GetCudaManager().GetDeviceCount();
    const ui32 elementsPerDevice = (sizesSum + deviceCount - 1) / deviceCount;
    TVector<TSlice> slices;
    slices.reserve(deviceCount);
    TSlice slice;
    for (ui32 i = 0, iEnd = sizes.size(), inSliceElementCount = 0, totalElementCount = 0; i < iEnd; ++i) {
        inSliceElementCount += sizes[i];
        totalElementCount += sizes[i];
        if (inSliceElementCount > elementsPerDevice) {
            slice.Right = totalElementCount;
            slices.push_back(slice);
            slice.Left = slice.Right;
            inSliceElementCount = 0;
        }
    }

    slice.Right = sizesSum;
    slices.push_back(slice);

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

static double RunOnGpu(
    const TData& data,
    const TMetricParams& params,
    const ui32 iterationCount)
{
    THPTimer timer;

    CATBOOST_INFO_LOG << "initializing data for GPU evaluator" << '\n';

    TVector<ui32> offsets;
    offsets.yresize(data.QueryCount);
    for (ui32 i = 0, iEnd = data.QueryCount, offset = 0; i < iEnd; offset += data.Sizes[i], (void)++i) {
        offsets[i] = offset;
    }

    TVector<float> weights;
    if (params.UseWeights) {
        weights.yresize(data.DocumentCount);
        for (ui32 i = 0, iEnd = data.QueryCount, offset = 0; i < iEnd; offset += data.Sizes[i], (void)++i) {
            const auto weight = data.Weights[i];
            for (ui32 j = 0, jEnd = data.Sizes[i]; j < jEnd; ++j) {
                weights[offset + j] = weight;
            }
        }
    } else {
        weights.assign(data.DocumentCount, 1.f);
    }

    CATBOOST_INFO_LOG << "initialized data for GPU evaluator in " << timer.Passed() << " sec." << '\n';

    CATBOOST_INFO_LOG << "uploading data to GPU devices" << '\n';
    timer.Reset();

    TDeviceRequestConfig cfg;
    auto cudaGuard = StartCudaManager(cfg, ELoggingLevel::Info);

    const auto groupsMapping = MakeDeviceAwareGroupsMapping(data.Sizes, data.DocumentCount);
    const auto elementsMapping = MakeDeviceAwareElementsMapping(data.Sizes, data.DocumentCount);
    auto deviceSizes = TStripeBuffer<ui32>::Create(groupsMapping);
    auto deviceBiasedOffsets = TStripeBuffer<ui32>::Create(groupsMapping);
    auto deviceOffsetsBias = MakeOffsetsBias(offsets, groupsMapping);
    auto deviceQueryWeights = TStripeBuffer<float>::Create(groupsMapping);
    auto deviceTargets = TStripeBuffer<float>::Create(elementsMapping);
    auto deviceApproxes = TStripeBuffer<float>::Create(elementsMapping);
    auto deviceWeights = TStripeBuffer<float>::Create(elementsMapping);

    deviceSizes.Write(data.Sizes);
    deviceBiasedOffsets.Write(offsets);
    deviceTargets.Write(data.Targets);
    deviceApproxes.Write(data.Approxes);
    deviceWeights.Write(weights);

    GetCudaManager().WaitComplete();

    CATBOOST_INFO_LOG << "uploaded data to GPU devices in " << timer.Passed() << " sec." << '\n';

    const auto description = TMetricParams::MakeDescription(params);

    double bestTime = Max<double>();
    for (ui32 iteration = 0; iteration < iterationCount; ++iteration) {
        CATBOOST_INFO_LOG << "gpu: " << LabeledOutput(description, iteration) << "; started" << '\n';
        timer.Reset();
        const auto ndcg = CalculateNdcg(
            deviceSizes.ConstCopyView(),
            deviceBiasedOffsets.ConstCopyView(),
            deviceOffsetsBias,
            deviceWeights.ConstCopyView(),
            deviceTargets.ConstCopyView(),
            deviceApproxes.ConstCopyView(),
            params.Type,
            {params.Top}).front();
        Y_FAKE_READ(ndcg);
        GatherBySizeAndOffset(
            deviceWeights,
            deviceSizes.ConstCopyView(),
            deviceBiasedOffsets.ConstCopyView(),
            deviceOffsetsBias,
            deviceQueryWeights,
            1);
        const auto queryWeightsSum = ReduceToHost(deviceQueryWeights);
        Y_FAKE_READ(queryWeightsSum);
        const auto iterationTime = timer.Passed();
        bestTime = Min(bestTime, iterationTime);
        CATBOOST_INFO_LOG << "gpu: " << LabeledOutput(description, iteration, iterationTime, bestTime) << '\n';
    }

    CATBOOST_INFO_LOG << "gpu: " << LabeledOutput(description, bestTime) << '\n';

    return bestTime;
}

static int Main(const TArgs& args) {
    CATBOOST_INFO_LOG << "generating random data for benchmark" << '\n';
    CATBOOST_INFO_LOG << LabeledOutput(args.DocumentCount, args.MaxDocumentsPerQuery, args.RelevanceScale, args.ApproxMultiplier) << '\n';
    THPTimer timer;
    const auto data = TData::Generate(args.RandomGeneratorSeed, args.MaxDocumentsPerQuery, args.DocumentCount, args.RelevanceScale, args.ApproxMultiplier);
    CATBOOST_INFO_LOG << "generated random data for benchmark in " << timer.Passed() << " sec." << '\n';
    CATBOOST_INFO_LOG << LabeledOutput(data.QueryCount) << '\n';

    TString delimiter;
    for (size_t i = 0; i < 80; ++i) {
        delimiter += '=';
    }

    TMap<TString, std::pair<bool, std::array<double, 2>>> gpuIsBetterByDescription;
    for (const auto type : {ENdcgMetricType::Base, ENdcgMetricType::Exp}) {
        for (const ui32 top : {3, 10, 30}) {
            for (const auto useWeights : {false, true}) {
                TMetricParams params;
                params.Type = type;
                params.Top = top;
                params.UseWeights = useWeights;
                const auto description = TMetricParams::MakeDescription(params);

                CATBOOST_INFO_LOG << "running benchmark on CPU" << '\n';
                timer.Reset();
                const auto cpuBestTime = RunOnCpu(
                    data,
                    args.CpuThreadCount,
                    params,
                    args.IterationCount);
                CATBOOST_INFO_LOG << "done running benchmark on CPU in " << timer.Passed() << " sec." << '\n';

                CATBOOST_INFO_LOG << "running benchmark on GPU" << '\n';
                timer.Reset();
                const auto gpuBestTime = RunOnGpu(
                    data,
                    params,
                    args.IterationCount);
                CATBOOST_INFO_LOG << "done running benchmark on GPU in " << timer.Passed() << " sec." << '\n';

                const auto bestTime = Min(cpuBestTime, gpuBestTime);
                const std::array<double, 2> dummy = {{bestTime, Max(cpuBestTime, gpuBestTime)}};
                gpuIsBetterByDescription[description] = std::make_pair(bestTime == gpuBestTime, dummy);

                CATBOOST_INFO_LOG << LabeledOutput(type, top, useWeights) << ";best time " << bestTime << " sec." << '\n';
                CATBOOST_INFO_LOG << "cpu best time: " << cpuBestTime << " sec." << '\n';
                CATBOOST_INFO_LOG << "gpu best time: " << gpuBestTime << " sec." << '\n';
                CATBOOST_INFO_LOG << (bestTime == gpuBestTime ? "gpu" : "cpu") << "is better" << '\n';
                CATBOOST_INFO_LOG << '\n' << delimiter << '\n' << '\n';
            }
        }
    }

    Cout << LabeledOutput(args.DocumentCount, data.QueryCount, args.MaxDocumentsPerQuery, args.CpuThreadCount) << '\n';
    for (const auto& [description, value] : gpuIsBetterByDescription) {
        const auto gpuIsBetter = value.first;
        const auto bestTime = value.second[0];
        const auto secondTime = value.second[1];
        Cout << LabeledOutput(description, gpuIsBetter, bestTime, secondTime / bestTime) << '\n';
    }

    return EXIT_SUCCESS;
}

int main(const int argc, const char* argv[]) {
    TSetLoggingVerbose verboseLogGuard;
    return Main(TArgs::ParseArgs(argc, argv));
}
