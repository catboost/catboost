#pragma once

#include "model.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/array_ref.h>
#include <util/generic/hash.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/stream/labeled.h>
#include <util/system/platform.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <algorithm>
#include <functional>
#include <limits>

#ifdef _sse2_
#include <emmintrin.h>
#endif

constexpr size_t FORMULA_EVALUATION_BLOCK_SIZE = 128;
constexpr ui32 MAX_VALUES_PER_BIN = 254;

inline void OneHotBinsFromTransposedCatFeatures(
    const TVector<TOneHotFeature>& OneHotFeatures,
    const THashMap<int, int> catFeaturePackedIndex,
    const size_t docCount,
    ui8*& result,
    TVector<ui32>& transposedHash
) {
    for (const auto& oheFeature : OneHotFeatures) {
        const auto catIdx = catFeaturePackedIndex.at(oheFeature.CatFeatureIndex);
        for (size_t docId = 0; docId < docCount; ++docId) {
            static_assert(sizeof(int) >= sizeof(i32));
            const int val = *reinterpret_cast<i32*>(&(transposedHash[catIdx * docCount + docId]));
            ui8* writePosition = &result[docId];
            for (size_t blockStart = 0;
                 blockStart < oheFeature.Values.size();
                 blockStart += MAX_VALUES_PER_BIN)
            {
                const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, oheFeature.Values.size());
                for (size_t borderIdx = blockStart; borderIdx < blockEnd; ++borderIdx) {
                    *writePosition |= (ui8)(val == oheFeature.Values[borderIdx]) * (borderIdx - blockStart + 1);
                }
                writePosition += docCount;
            }
        }
        result += docCount * ((oheFeature.Values.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
    }
}

template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
Y_FORCE_INLINE void BinarizeFloatsNonSse(
        const size_t docCount,
        TFloatFeatureAccessor floatAccessor,
        const TConstArrayRef<float> borders,
        size_t start,
        ui8*& result,
        float nanSubstitutionValue=0.0f
) {
    const auto docCount8 = (docCount | 0x7) ^ 0x7;
    for (size_t docId = 0; docId < docCount8; docId += 8) {
        float val[8] = {
            floatAccessor(start + docId + 0),
            floatAccessor(start + docId + 1),
            floatAccessor(start + docId + 2),
            floatAccessor(start + docId + 3),
            floatAccessor(start + docId + 4),
            floatAccessor(start + docId + 5),
            floatAccessor(start + docId + 6),
            floatAccessor(start + docId + 7)
        };
        if (UseNanSubstitution) {
            for (size_t i = 0; i < 8; ++i) {
                if (IsNan(val[i])) {
                    val[i] = nanSubstitutionValue;
                }
            }
        }
        ui32* writePtr = (ui32*)(result + docId);
        for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
            const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
            for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                const auto border = borders[borderId];
                writePtr[0] += (val[0] > border) + ((val[1] > border) << 8) + ((val[2] > border) << 16)
                    + ((val[3] > border) << 24);
                writePtr[1] += (val[4] > border) + ((val[5] > border) << 8) + ((val[6] > border) << 16)
                    + ((val[7] > border) << 24);
            }
            writePtr = (ui32*)((ui8*)writePtr + docCount);
        }
    }
    for (size_t docId = docCount8; docId < docCount; ++docId) {
        float val = floatAccessor(start + docId);
        if (UseNanSubstitution) {
            if (IsNan(val)) {
                val = nanSubstitutionValue;
            }
        }

        ui8* writePtr = result + docId;
        for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
            const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
            for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                *writePtr += (ui8) (val > borders[borderId]);
            }
            writePtr += docCount;
        }
    }
    result += docCount * ((borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
}

#ifndef _sse2_

template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
Y_FORCE_INLINE void BinarizeFloats(
    const size_t docCount,
    TFloatFeatureAccessor floatAccessor,
    const TConstArrayRef<float> borders,
    size_t start,
    ui8*& result,
    const float nanSubstitutionValue = 0.0f
) {
    BinarizeFloatsNonSse<UseNanSubstitution, TFloatFeatureAccessor>(
        docCount,
        floatAccessor,
        borders,
        start,
        result,
        nanSubstitutionValue
    );
};

#else

template <bool UseNanSubstitution, typename TFloatFeatureAccessor>
Y_FORCE_INLINE void BinarizeFloats(
    const size_t docCount,
    TFloatFeatureAccessor floatAccessor,
    const TConstArrayRef<float> borders,
    size_t start,
    ui8*& result,
    const float nanSubstitutionValue = 0.0f
) {
    const __m128 substitutionValVec = _mm_set1_ps(nanSubstitutionValue);
    const auto docCount16 = (docCount | 0xf) ^ 0xf;
    for (size_t docId = 0; docId < docCount16; docId += 16) {
        const float val[16] = {
            floatAccessor(start + docId + 0),
            floatAccessor(start + docId + 1),
            floatAccessor(start + docId + 2),
            floatAccessor(start + docId + 3),
            floatAccessor(start + docId + 4),
            floatAccessor(start + docId + 5),
            floatAccessor(start + docId + 6),
            floatAccessor(start + docId + 7),
            floatAccessor(start + docId + 8),
            floatAccessor(start + docId + 9),
            floatAccessor(start + docId + 10),
            floatAccessor(start + docId + 11),
            floatAccessor(start + docId + 12),
            floatAccessor(start + docId + 13),
            floatAccessor(start + docId + 14),
            floatAccessor(start + docId + 15)
        };
        const __m128i mask = _mm_set1_epi8(1);
        __m128 floats0 = _mm_load_ps(val);
        __m128 floats1 = _mm_load_ps(val + 4);
        __m128 floats2 = _mm_load_ps(val + 8);
        __m128 floats3 = _mm_load_ps(val + 12);

        if (UseNanSubstitution) {
            {
                const __m128 masks = _mm_cmpunord_ps(floats0, floats0);
                floats0 = _mm_or_ps(_mm_andnot_ps(masks, floats0), _mm_and_ps(masks, substitutionValVec));
            }
            {
                const __m128 masks = _mm_cmpunord_ps(floats1, floats1);
                floats1 = _mm_or_ps(_mm_andnot_ps(masks, floats1), _mm_and_ps(masks, substitutionValVec));
            }
            {
                const __m128 masks = _mm_cmpunord_ps(floats2, floats2);
                floats2 = _mm_or_ps(_mm_andnot_ps(masks, floats2), _mm_and_ps(masks, substitutionValVec));
            }
            {
                const __m128 masks = _mm_cmpunord_ps(floats3, floats3);
                floats3 = _mm_or_ps(_mm_andnot_ps(masks, floats3), _mm_and_ps(masks, substitutionValVec));
            }
        }
        ui8* writePtr = result + docId;
        for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
            __m128i resultVec = _mm_setzero_si128();
            const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
            for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                const __m128 borderVec = _mm_set1_ps(borders[borderId]);
                const __m128i r0 = _mm_castps_si128(_mm_cmpgt_ps(floats0, borderVec));
                const __m128i r1 = _mm_castps_si128(_mm_cmpgt_ps(floats1, borderVec));
                const __m128i r2 = _mm_castps_si128(_mm_cmpgt_ps(floats2, borderVec));
                const __m128i r3 = _mm_castps_si128(_mm_cmpgt_ps(floats3, borderVec));
                const __m128i packed = _mm_packs_epi16(_mm_packs_epi32(r0, r1), _mm_packs_epi32(r2, r3));
                resultVec = _mm_add_epi8(resultVec, _mm_and_si128(packed, mask));
            }
            _mm_storeu_si128((__m128i *)writePtr, resultVec);
            writePtr += docCount;
        }
    }
    for (size_t docId = docCount16; docId < docCount; ++docId) {
        float val = floatAccessor(start + docId);
        if (UseNanSubstitution) {
            if (IsNan(val)) {
                val = nanSubstitutionValue;
            }
        }
        ui8* writePtr = result + docId;
        for (size_t blockStart = 0; blockStart < borders.size(); blockStart += MAX_VALUES_PER_BIN) {
            const size_t blockEnd = Min(blockStart + MAX_VALUES_PER_BIN, borders.size());
            for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                *writePtr += (ui8)(val > borders[borderId]);
            }
            writePtr += docCount;
        }
    }
    result += docCount * ((borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
}

#endif

/**
* This function binarizes
*/
template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
inline void BinarizeFeatures(
    const TFullModel& model,
    TFloatFeatureAccessor floatAccessor,
    TCatFeatureAccessor catFeatureAccessor,
    size_t start,
    size_t end,
    TArrayRef<ui8> result,
    TVector<ui32>& transposedHash,
    TVector<float>& ctrs
) {
    const auto docCount = end - start;
    ui8* resultPtr = result.data();
    std::fill(result.begin(), result.end(), 0);
    for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        if (!floatFeature.HasNans || floatFeature.NanValueTreatment == NCatBoostFbs::ENanValueTreatment_AsIs) {
            BinarizeFloats<false>(
                docCount,
                [&floatFeature, floatAccessor](size_t index) { return floatAccessor(floatFeature, index); },
                floatFeature.Borders,
                start,
                resultPtr
            );
        } else {
            const float infinity = std::numeric_limits<float>::infinity();
            if (floatFeature.NanValueTreatment == NCatBoostFbs::ENanValueTreatment_AsFalse) {
                BinarizeFloats<true>(
                    docCount,
                    [&floatFeature, floatAccessor](size_t index) { return floatAccessor(floatFeature, index); },
                    floatFeature.Borders,
                    start,
                    resultPtr,
                    -infinity
                );
            } else {
                Y_ASSERT(floatFeature.NanValueTreatment == NCatBoostFbs::ENanValueTreatment_AsTrue);
                BinarizeFloats<true>(
                    docCount,
                    [&floatFeature, floatAccessor](size_t index) { return floatAccessor(floatFeature, index); },
                    floatFeature.Borders,
                    start,
                    resultPtr,
                    infinity
                );
            }
        }
    }
    if (model.HasCategoricalFeatures()) {
        THashMap<int, int> catFeaturePackedIndexes;
        int usedFeatureIdx = 0;
        for (const auto& catFeature : model.ObliviousTrees.CatFeatures) {
            if (!catFeature.UsedInModel) {
                continue;
            }
            catFeaturePackedIndexes[catFeature.FeatureIndex] = usedFeatureIdx;
            for (size_t docId = 0, writeIdx = usedFeatureIdx * docCount;
                 docId < docCount;
                 ++docId, ++writeIdx)
            {
                transposedHash[writeIdx] = catFeatureAccessor(catFeature, start + docId);
            }
            ++usedFeatureIdx;
        }
        Y_ASSERT(model.GetUsedCatFeaturesCount() == (size_t)usedFeatureIdx);
        OneHotBinsFromTransposedCatFeatures(
            model.ObliviousTrees.OneHotFeatures,
            catFeaturePackedIndexes,
            docCount,
            resultPtr,
            transposedHash
        );
        if (!model.ObliviousTrees.GetUsedModelCtrs().empty()) {
            model.CtrProvider->CalcCtrs(
                model.ObliviousTrees.GetUsedModelCtrs(),
                result,
                transposedHash,
                docCount,
                ctrs
            );
        }
        for (size_t i = 0; i < model.ObliviousTrees.CtrFeatures.size(); ++i) {
            const auto& ctr = model.ObliviousTrees.CtrFeatures[i];
            auto ctrFloatsPtr = &ctrs[i * docCount];
            BinarizeFloats<false>(
                docCount,
                [ctrFloatsPtr](size_t index) { return ctrFloatsPtr[index]; },
                ctr.Borders,
                0,
                resultPtr
            );
        }
    }
}

/**
* This function is for quantized pool
*/
template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
inline void AssignFeatureBins(
    const TFullModel& model,
    TFloatFeatureAccessor floatAccessor,
    TCatFeatureAccessor /*catAccessor*/,
    size_t start,
    size_t end,
    TArrayRef<ui8> result
) {
    CB_ENSURE(!model.HasCategoricalFeatures(), "Quantized datasets with categorical features are not currently supported");
    ui8* resultPtr = result.data();
    for (const auto& floatFeature : model.ObliviousTrees.FloatFeatures) {
        if (!floatFeature.UsedInModel()) {
            continue;
        }
        for (ui32 docId = start; docId < end; ++docId) {
            *resultPtr = floatAccessor(floatFeature, docId);
            resultPtr++;
        }
    }
}

using TCalcerIndexType = ui32;

using TTreeCalcFunction = std::function<void(
    const TFullModel& model,
    const ui8* __restrict binFeatures,
    size_t docCountInBlock,
    TCalcerIndexType* __restrict indexesVec,
    size_t treeStart,
    size_t treeEnd,
    double* __restrict results)>;

void CalcIndexes(
    bool needXorMask,
    const ui8* __restrict binFeatures,
    size_t docCountInBlock,
    ui32* __restrict indexesVec,
    const TRepackedBin* __restrict treeSplitsCurPtr,
    int curTreeSize);

TTreeCalcFunction GetCalcTreesFunction(const TFullModel& model, size_t docCountInBlock);

template <class X>
inline X* GetAligned(X* val) {
    uintptr_t off = ((uintptr_t)val) & 0xf;
    val = (X *)((ui8 *)val - off + 0x10);
    return val;
}

template <bool isQuantizedFeaturesData = false, typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
inline void CalcGeneric(
    const TFullModel& model,
    TFloatFeatureAccessor floatFeatureAccessor,
    TCatFeatureAccessor catFeaturesAccessor,
    size_t docCount,
    size_t treeStart,
    size_t treeEnd,
    TArrayRef<double> results
) {
    size_t blockSize = FORMULA_EVALUATION_BLOCK_SIZE;
    blockSize = Min(blockSize, docCount);
    const size_t binSlots = blockSize * model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount();
    TArrayRef<ui8> binFeatures;
    TVector<ui8> binFeaturesHolder;
    if (binSlots < 65536) { // 65KB of stack maximum
        binFeatures = MakeArrayRef(GetAligned((ui8*)(alloca(binSlots + 0x20))), binSlots);
    } else {
        binFeaturesHolder.yresize(blockSize * model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount());
        binFeatures = binFeaturesHolder;
    }
    auto calcTrees = GetCalcTreesFunction(model, blockSize);

    CB_ENSURE(
        results.size() == docCount * model.ObliviousTrees.ApproxDimension,
        "`results` size is insufficient: "
        LabeledOutput(results.size(), docCount * model.ObliviousTrees.ApproxDimension));
    std::fill(results.begin(), results.end(), 0.0);
    TVector<TCalcerIndexType> indexesVec(blockSize);
    TVector<ui32> transposedHash(blockSize * model.GetUsedCatFeaturesCount());
    TVector<float> ctrs(model.ObliviousTrees.GetUsedModelCtrs().size() * blockSize);
    for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
        const auto docCountInBlock = Min(blockSize, docCount - blockStart);
        if constexpr (!isQuantizedFeaturesData) {
            BinarizeFeatures(
                model,
                floatFeatureAccessor,
                catFeaturesAccessor,
                blockStart,
                blockStart + docCountInBlock,
                binFeatures,
                transposedHash,
                ctrs
            );
        } else {
            AssignFeatureBins(
                model,
                floatFeatureAccessor,
                catFeaturesAccessor,
                blockStart,
                blockStart + docCountInBlock,
                binFeatures
            );
        }

        calcTrees(
            model,
            binFeatures.data(),
            docCountInBlock,
            docCount == 1 ? nullptr : indexesVec.data(),
            treeStart,
            treeEnd,
            results.data() + blockStart * model.ObliviousTrees.ApproxDimension
        );
    }
}


/**
 * Warning: use aggressive caching. Stores all binarized features in RAM
 */
class TFeatureCachedTreeEvaluator {
public:
    template <typename TFloatFeatureAccessor,
             typename TCatFeatureAccessor>
    TFeatureCachedTreeEvaluator(
        const TFullModel& model,
        TFloatFeatureAccessor floatFeatureAccessor,
        TCatFeatureAccessor catFeaturesAccessor,
        size_t docCount
    )
        : Model(model)
        , DocCount(docCount)
    {
        size_t blockSize = FORMULA_EVALUATION_BLOCK_SIZE;
        BlockSize = Min(blockSize, docCount);
        CalcFunction = GetCalcTreesFunction(Model, BlockSize);
        TVector<ui32> transposedHash(blockSize * model.GetUsedCatFeaturesCount());
        TVector<float> ctrs(model.ObliviousTrees.GetUsedModelCtrs().size() * blockSize);
        {
            for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
                const auto docCountInBlock = Min(blockSize, docCount - blockStart);
                TVector<ui8> binFeatures(
                    model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() * blockSize);
                BinarizeFeatures(
                    model,
                    floatFeatureAccessor,
                    catFeaturesAccessor,
                    blockStart,
                    blockStart + docCountInBlock,
                    binFeatures,
                    transposedHash,
                    ctrs
                );
                BinFeatures.push_back(std::move(binFeatures));
            }
        }
    }

    template <typename TFloatFeatureAccessor>
    TFeatureCachedTreeEvaluator(
        const TFullModel& model,
        TFloatFeatureAccessor floatFeatureAccessor,
        size_t docCount
    )
        : Model(model)
        , DocCount(docCount)
    {
        size_t blockSize = FORMULA_EVALUATION_BLOCK_SIZE;
        BlockSize = Min(blockSize, docCount);
        CalcFunction = GetCalcTreesFunction(Model, BlockSize);
        for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
            const auto docCountInBlock = Min(blockSize, docCount - blockStart);
            TVector<ui8> binFeatures(
                model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() * blockSize);
            AssignFeatureBins(
                model,
                floatFeatureAccessor,
                nullptr,
                blockStart,
                blockStart + docCountInBlock,
                binFeatures
            );
            BinFeatures.push_back(std::move(binFeatures));
        }
    }

    void Calc(size_t treeStart, size_t treeEnd, TArrayRef<double> results) const;
private:
    const TFullModel& Model;
    TVector<TVector<ui8>> BinFeatures;
    TTreeCalcFunction CalcFunction;
    ui64 DocCount;
    ui64 BlockSize;
};

template <typename TFloatFeatureAccessor, typename TCatFeatureAccessor>
inline TVector<TVector<double>> CalcTreeIntervalsGeneric(
    const TFullModel& model,
    TFloatFeatureAccessor floatFeatureAccessor,
    TCatFeatureAccessor catFeaturesAccessor,
    size_t docCount,
    size_t incrementStep)
{
    size_t blockSize = FORMULA_EVALUATION_BLOCK_SIZE;
    blockSize = Min(blockSize, docCount);
    auto treeStepCount = (model.ObliviousTrees.TreeSizes.size() + incrementStep - 1) / incrementStep;
    TVector<TVector<double>> results(docCount, TVector<double>(treeStepCount));
    CB_ENSURE(model.ObliviousTrees.ApproxDimension == 1);
    TVector<ui8> binFeatures(model.ObliviousTrees.GetEffectiveBinaryFeaturesBucketsCount() * blockSize);
    TVector<TCalcerIndexType> indexesVec(blockSize);
    TVector<ui32> transposedHash(blockSize * model.GetUsedCatFeaturesCount());
    TVector<float> ctrs(model.ObliviousTrees.GetUsedModelCtrs().size() * blockSize);
    TVector<double> tmpResult(docCount);
    TArrayRef<double> tmpResultRef(tmpResult);
    auto calcTrees = GetCalcTreesFunction(model, blockSize);
    for (size_t blockStart = 0; blockStart < docCount; blockStart += blockSize) {
        const auto docCountInBlock = Min(blockSize, docCount - blockStart);
        BinarizeFeatures(
            model,
            floatFeatureAccessor,
            catFeaturesAccessor,
            blockStart,
            blockStart + docCountInBlock,
            binFeatures,
            transposedHash,
            ctrs
        );
        for (size_t stepIdx = 0; stepIdx < treeStepCount; ++stepIdx) {
            calcTrees(
                model,
                binFeatures.data(),
                docCountInBlock,
                indexesVec.data(),
                stepIdx * incrementStep,
                Min((stepIdx + 1) * incrementStep, model.ObliviousTrees.TreeSizes.size()),
                tmpResultRef.data() + blockStart * model.ObliviousTrees.ApproxDimension
            );
            for (size_t i = 0; i < docCountInBlock; ++i) {
                results[blockStart + i][stepIdx] = tmpResult[i];
            }
        }
    }
    return results;
}
