#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/math_utils.h>
#include <catboost/private/libs/data_types/query.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/fast_exp/fast_exp.h>
#include <library/cpp/fast_log/fast_log.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>
#include <util/system/yassert.h>


template <bool StoreExpApprox>
static inline double UpdateApprox(double approx, double approxDelta) {
    return StoreExpApprox ? approx * approxDelta : approx + approxDelta;
}

static inline double UpdateApprox(bool storeExpApprox, double approx, double approxDelta) {
    return storeExpApprox ? approx * approxDelta : approx + approxDelta;
}

template <bool StoreExpApprox>
static inline double ApplyLearningRate(double approxDelta, double learningRate) {
    return StoreExpApprox ? fast_exp(FastLogf(approxDelta) * learningRate) : approxDelta * learningRate;
}

static inline double GetNeutralApprox(bool storeExpApproxes) {
    if (storeExpApproxes) {
        return 1.0;
    } else {
        return 0.0;
    }
}

static inline double ExpApproxIf(bool storeExpApproxes, double approx) {
    return storeExpApproxes ? fast_exp(approx) : approx;
}

static inline void ExpApproxIf(bool storeExpApproxes, TArrayRef<double> approx) {
    if (storeExpApproxes) {
        NCB::FastExpWithInfInplace(approx.data(), approx.size());
    }
}

static inline void ExpApproxIf(bool storeExpApproxes, TVector<TVector<double>>* approxMulti) {
    for (auto& approx : *approxMulti) {
        ExpApproxIf(storeExpApproxes, approx);
    }
}


static inline bool IsStoreExpApprox(ELossFunction lossFunction) {
    return EqualToOneOf(
        lossFunction,
        ELossFunction::Logloss,
        ELossFunction::LogLinQuantile,
        ELossFunction::Poisson,
        ELossFunction::CrossEntropy,
        ELossFunction::PairLogit,
        ELossFunction::PairLogitPairwise,
        ELossFunction::YetiRank,
        ELossFunction::YetiRankPairwise
    );
}

inline void CalcPairwiseWeights(
    const TVector<TQueryInfo>& queriesInfo,
    int queriesCount,
    TVector<float>* pairwiseWeights
) {
    Fill(pairwiseWeights->begin(), pairwiseWeights->end(), 0);
    for (int queryIndex = 0; queryIndex < queriesCount; ++queryIndex) {
        const auto& queryInfo = queriesInfo[queryIndex];
        for (int docId = 0; docId < queryInfo.Competitors.ysize(); ++docId) {
            for (const auto& competitor : queryInfo.Competitors[docId]) {
                (*pairwiseWeights)[queryInfo.Begin + docId] += competitor.Weight;
                (*pairwiseWeights)[queryInfo.Begin + competitor.Id] += competitor.Weight;
            }
        }
    }
}

inline ui32 AdjustBlockSize(ui32 objectCount, ui32 regularBlockSize) {
    return (objectCount < 10000) ? 10000 : regularBlockSize;
}

inline ui32 AdjustBlockCountLimit(ui32 objectCount, ui32 regularBlockCount) {
    return (objectCount < 10000) ? 1 : regularBlockCount;
}

template <typename TUpdateFunc>
inline void UpdateApprox(
    const TUpdateFunc& updateFunc,
    const TVector<TVector<double>>& delta,
    TVector<TVector<double>>* approx,
    NPar::ILocalExecutor* localExecutor
) {
    Y_ASSERT(delta.size() == approx->size());
    for (size_t dimensionIdx : xrange(delta.size())) {
        TConstArrayRef<double> deltaDim(delta[dimensionIdx]);

        // deltaDim.size() < approxDim.size(), if delta is leaf values
        TArrayRef<double> approxDim((*approx)[dimensionIdx]);
        if (approxDim.empty()) {
            continue;
        }

        NPar::ILocalExecutor::TExecRangeParams blockParams(0, approxDim.size());
        blockParams.SetBlockCount(AdjustBlockCountLimit(approxDim.size(), localExecutor->GetThreadCount() + 1));
        localExecutor->ExecRange(
            [=, &updateFunc](int idx) {
                updateFunc(deltaDim, approxDim, idx);
            },
            blockParams,
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

inline void CopyApprox(
    const TVector<TVector<double>>& src,
    TVector<TVector<double>>* dst,
    NPar::ILocalExecutor* localExecutor
) {
    if (dst->empty() && !src.empty()) {
        dst->resize(src.size());
        const auto rowSize = src[0].size();
        for (auto& row : *dst) {
            row.yresize(rowSize);
        }
    }
    const auto copyFunc
        = [] (TConstArrayRef<double> src, TArrayRef<double> dst, size_t idx) { dst[idx] = src[idx]; };
    UpdateApprox(copyFunc, src, dst, localExecutor);
}

void NormalizeLeafValues(
    bool isPairwise,
    double learningRate,
    const TVector<double>& leafWeightsSum,
    TVector<TVector<double>>* treeValues
);

TVector<double> SumLeafWeights(
    size_t leafCount,
    TConstArrayRef<TIndexType> leafIndices,
    TConstArrayRef<ui32> learnPermutation,
    TConstArrayRef<float> learnWeights, // can be empty
    NPar::ILocalExecutor* localExecutor
);

template <typename TElementType>
inline void AddElementwise(const TVector<TElementType>& value, TVector<TElementType>* accumulator) {
    Y_ASSERT(value.size() == accumulator->size());
    for (int idx : xrange(value.size())) {
        AddElementwise(value[idx], &(*accumulator)[idx]);
    }
}

template <>
inline void AddElementwise<double>(const TVector<double>& value, TVector<double>* accumulator) {
    Y_ASSERT(value.size() == accumulator->size());
    for (int idx : xrange(value.size())) {
        (*accumulator)[idx] += value[idx];
    }
}

template <typename TElementType>
inline void AddElementwise(TConstArrayRef<TElementType> value, TVector<TElementType>* accumulator) {
    Y_ASSERT(value.size() == accumulator->size());
    for (int idx : xrange(value.size())) {
        AddElementwise(value[idx], &(*accumulator)[idx]);
    }
}

template <>
inline void AddElementwise<double>(TConstArrayRef<double> value, TVector<double>* accumulator) {
    Y_ASSERT(value.size() == accumulator->size());
    for (int idx : xrange(value.size())) {
        (*accumulator)[idx] += value[idx];
    }
}

template <typename TElementType>
inline TVector<TElementType> ScaleElementwise(double scale, const TVector<TElementType>& value) {
    TVector<TElementType> scaledValue(value);
    for (int idx : xrange(value.size())) {
        scaledValue[idx] = ScaleElementwise(scale, value[idx]);
    }
    return scaledValue;
}

template <>
inline TVector<double> ScaleElementwise<double>(double scale, const TVector<double>& value) {
    TVector<double> scaledValue(value);
    for (int idx : xrange(value.size())) {
        scaledValue[idx] = value[idx] * scale;
    }
    return scaledValue;
}


template <class T>
void InitApproxFromBaseline(
    const ui32 endIdx,
    TConstArrayRef<TConstArrayRef<T>> baseline,
    TConstArrayRef<ui32> learnPermutation,
    bool storeExpApproxes,
    TVector<TVector<double>>* approx
) {
    const ui32 learnSampleCount = learnPermutation.size();
    const int approxDimension = approx->ysize();
    for (int dim = 0; dim < approxDimension; ++dim) {
        for (ui32 docId : xrange(endIdx)) {
            ui32 initialIdx = docId;
            if (docId < learnSampleCount) {
                initialIdx = learnPermutation[docId];
            }
            (*approx)[dim][docId] = baseline[dim][initialIdx];
        }
        ExpApproxIf(
            storeExpApproxes,
            TArrayRef<double>((*approx)[dim].data(), (*approx)[dim].data() + endIdx)
        );
    }
}

void InitApproxes(
    int size,
    const TMaybe<TVector<double>>& startingApprox,
    double approxDimension,
    bool storeExpApproxes,
    TVector<TVector<double>>* approx);
