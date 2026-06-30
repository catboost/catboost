#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>
#include <catboost/cuda/targets/kernel.h>
#include <catboost/private/libs/algo_helpers/ders_holder.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/libs/metrics/sample.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>
#include <util/system/info.h>

using TVec = TSingleBuffer<float>;
using Derivatives = std::pair<TVector<float>, TVector<float>>;
using CpuResult = std::pair<float, TVector<TDers>>;
using GpuResult = std::pair<float, Derivatives>;

static inline void GenerateSamples(TRandom & random,
                                   ui64 size,
                                   TVector<float>& classes,
                                   TVector<TVector<float>>& predictions,
                                   TVector<float>& weights,
                                   double rate = 0.6) {
    classes.resize(size);
    predictions[0].resize(size);
    weights.resize(size, 1.0f);

    double positiveApproxMean = 1;
    double negativeApproxMean = -1;

    for (ui64 i = 0; i < size; ++i) {
        classes[i] = random.NextUniformL() % 2;

        const double classifierClass = (random.NextUniform() < rate) ? classes[i] : 1.0 - classes[i];

        double mean = classifierClass ? positiveApproxMean : negativeApproxMean;
        predictions[0][i] = random.NextGaussian() + mean;
    }
}

static inline CpuResult CalculateLossAndDerivativesOnCpu(TVector<float>& targets,
                                                         TVector<TVector<float>>& cursor,
                                                         TVector<float>& weights,
                                                         const ELossFunction& lossFunction,
                                                         const IDerCalcer& error,
                                                         double param,
                                                         TString paramName) {

    const auto metric = std::move(CreateSingleTargetMetric(lossFunction,
                                                           TLossParams::FromVector({{paramName, ToString(param)}}),
                                                           /*approxDimension=*/1)[0]);
    NPar::TLocalExecutor executor;

    TVector<TVector<double>> approxes(1, TVector<double>(cursor[0].size()));
    for (ui32 index = 0; index < cursor[0].size(); ++index) {
        approxes[0][index] = static_cast<double>(cursor[0][index]);
    }

    TMetricHolder score = metric->Eval(approxes, targets, weights, {}, 0, targets.size(), executor);

    const ui32 objectsCount = targets.size();
    TVector<TDers> derivatives(objectsCount);
    error.CalcDersRange(
            /*start=*/0,
                      objectsCount,
            /*calcThirdDer=*/false,
                      approxes[0].data(),
            /*approxDeltas=*/nullptr,
                      targets.data(),
                      weights.data(),
                      derivatives.data()
    );

    return std::make_pair(metric->GetFinalError(score), std::move(derivatives));
}

static inline std::pair<float, Derivatives> CalculateLossAndDerivativesOnGpu(TVector<float>& targets,
                                                                             TVector<TVector<float>>& cursor,
                                                                             TVector<float>& weights,
                                                                             const ELossFunction& lossFunction,
                                                                             double param) {
    auto docsMapping = NCudaLib::TSingleMapping(0, targets.size());
    auto targetsGpu = [&]() {
        auto tmp = TVec::Create(docsMapping);
        tmp.Write(targets);
        return tmp.ConstCopyView();
    }();

    auto weightsGpu = [&]() {
        auto tmp = TVec::Create(docsMapping);
        tmp.Write(weights);
        return tmp.ConstCopyView();
    }();

    auto cursorGpu = [&]() {
        auto tmp = TVec::Create(docsMapping);
        tmp.Write(cursor[0]);
        return tmp.ConstCopyView();
    }();

    auto tmp = TVec::Create(cursorGpu.GetMapping().RepeatOnAllDevices(1));
    auto der = TVec::CopyMapping(cursorGpu);
    auto der2 = TVec::CopyMapping(cursorGpu);
    ApproximatePointwise(targetsGpu,
                         weightsGpu,
                         cursorGpu,
                         lossFunction,
                         param,
                         &tmp,
                         &der,
                         &der2);

    //
    TVector<float> derVec;
    TVector<float> der2Vec;
    der.Read(derVec);
    der2.Read(der2Vec);
    //

    Derivatives derivatives = std::make_pair(derVec, der2Vec);
    const float score = -ReadReduce(tmp)[0] / targets.size();
    return std::make_pair(score, derivatives);
}

static inline void TestLossFunctionImpl(ui64 seed,
                                        double param,
                                        TString paramName,
                                        const ELossFunction& lossFunction,
                                        const IDerCalcer& error) {
    constexpr ui32 SIZE = 100;

    TRandom random(seed);

    auto stopCudaManagerGuard = StartCudaManager();
    {
        TVector<float> targets;
        TVector<TVector<float>> cursor(1);
        TVector<float> weights;

        GenerateSamples(random, SIZE, targets, cursor, weights);

        //
        CpuResult cpuResult = CalculateLossAndDerivativesOnCpu(targets, cursor, weights, lossFunction, error, param, paramName);
        GpuResult gpuResult = CalculateLossAndDerivativesOnGpu(targets, cursor, weights, lossFunction, param);
        //

        const float PRECISION = 1e-4f;
        UNIT_ASSERT_DOUBLES_EQUAL(cpuResult.first, gpuResult.first, PRECISION);
        for (ui32 index = 0; index < targets.size(); ++index) {
            UNIT_ASSERT_DOUBLES_EQUAL(cpuResult.second[index].Der1, gpuResult.second.first[index], PRECISION);
            UNIT_ASSERT_DOUBLES_EQUAL(cpuResult.second[index].Der2, -gpuResult.second.second[index], PRECISION); // 2nd der signs are opposite
        }
    }
}
