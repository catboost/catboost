#include "quantized_features_info.h"

#include "feature_index.h"
#include "sparse_columns.h"

#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/helpers/dbg_output.h>
#include <catboost/libs/helpers/serialization.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <library/cpp/dbg_output/dump.h>

#include <util/generic/cast.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/utility.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>


inline int operator&(NCatboostOptions::TBinarizationOptions& binarizationOptions, IBinSaver& binSaver) {
    EBorderSelectionType borderSelectionType;
    ui32 borderCount;
    ENanMode nanMode;
    if (!binSaver.IsReading()) {
        borderSelectionType = binarizationOptions.BorderSelectionType;
        borderCount = binarizationOptions.BorderCount;
        nanMode = binarizationOptions.NanMode;
    }
    binSaver.AddMulti(borderSelectionType, borderCount, nanMode);
    if (binSaver.IsReading()) {
        binarizationOptions.BorderSelectionType = borderSelectionType;
        binarizationOptions.BorderCount = borderCount;
        binarizationOptions.NanMode = nanMode;
    }
    return 0;
}


namespace NCB {
    static bool ApproximatelyEqualQuantization(
        const TMap<ui32, TQuantizationWithSerialization>& lhs,
        const TMap<ui32, TQuantizationWithSerialization>& rhs
    ) {
        constexpr auto EPS = 1.e-6f;

        if (lhs.size() != rhs.size()) {
            return false;
        }

        for (const auto& featureIdxAndValue : lhs) {
            auto* rhsValue = MapFindPtr(rhs, featureIdxAndValue.first);
            if (!rhsValue) {
                return false;
            }
            if (!ApproximatelyEqual<float>(featureIdxAndValue.second.Borders, rhsValue->Borders, EPS)) {
                return false;
            }
        }
        for (const auto& featureIdxAndValue : rhs) {
            if (!lhs.contains(featureIdxAndValue.first)) {
                return false;
            }
        }
        return true;
    }

    static bool EqualNanModesWithDefault(
        const TMap<ui32, ENanMode>& lhs,
        const TMap<ui32, ENanMode>& rhs
    ) {
        auto lhsIter = lhs.begin();
        auto rhsIter = rhs.begin();

        while ((lhsIter != lhs.end()) && (rhsIter != rhs.end())) {
            if (lhsIter->first < rhsIter->first) {
                if (lhsIter->second != ENanMode::Forbidden) {
                    return false;
                }
                ++lhsIter;
            } else if (lhsIter->first > rhsIter->first) {
                if (rhsIter->second != ENanMode::Forbidden) {
                    return false;
                }
                ++rhsIter;
            } else { // lhsIter->first == rhsIter->first
                if (lhsIter->second != rhsIter->second) {
                    return false;
                }
                ++lhsIter;
                ++rhsIter;
            }
        }
        for (; lhsIter != lhs.end(); ++lhsIter) {
            if (lhsIter->second != ENanMode::Forbidden) {
                return false;
            }
        }
        for (; rhsIter != rhs.end(); ++rhsIter) {
            if (rhsIter->second != ENanMode::Forbidden) {
                return false;
            }
        }
        return true;
    }

    TQuantizedFeaturesInfo::TQuantizedFeaturesInfo(
        const NCB::TFeaturesLayout &featuresLayout,
        TConstArrayRef<ui32> ignoredFeatures,
        NCatboostOptions::TBinarizationOptions commonFloatFeaturesBinarization,
        TMap<ui32, NCatboostOptions::TBinarizationOptions> perFloatFeatureQuantization,
        bool floatFeaturesAllowNansInTestOnly)
        : TQuantizedFeaturesInfo(
            featuresLayout,
            ignoredFeatures,
            commonFloatFeaturesBinarization,
            perFloatFeatureQuantization,
            NCatboostOptions::TTextProcessingOptions(),
            NCatboostOptions::TEmbeddingProcessingOptions(),
            floatFeaturesAllowNansInTestOnly
        )
    {}

    static TVector<ui32> GetAvailableTextFeatureIndices(const TFeaturesLayout& featuresLayout) {
        TVector<ui32> textFeatureIndices;
        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Text>(
            [&textFeatureIndices](TTextFeatureIdx textFeatureIdx) {
                textFeatureIndices.push_back(textFeatureIdx.Idx);
            }
        );
        return textFeatureIndices;
    }

    static TVector<ui32> GetAvailableEmbeddingFeatureIndices(const TFeaturesLayout& featuresLayout) {
        TVector<ui32> embeddingFeatureIndices;
        featuresLayout.IterateOverAvailableFeatures<EFeatureType::Embedding>(
            [&embeddingFeatureIndices](TEmbeddingFeatureIdx embeddingFeatureIdx) {
                embeddingFeatureIndices.push_back(embeddingFeatureIdx.Idx);
            }
        );
        return embeddingFeatureIndices;
    }

    TQuantizedFeaturesInfo::TQuantizedFeaturesInfo(
        const TFeaturesLayout& featuresLayout,
        TConstArrayRef<ui32> ignoredFeatures,
        NCatboostOptions::TBinarizationOptions commonFloatFeaturesBinarization,
        TMap<ui32, NCatboostOptions::TBinarizationOptions> perFloatFeatureQuantization,
        const NCatboostOptions::TTextProcessingOptions& textFeaturesProcessing,
        const NCatboostOptions::TEmbeddingProcessingOptions& embeddingProcessingOptions,
        bool floatFeaturesAllowNansInTestOnly)
        : FeaturesLayout(MakeIntrusive<TFeaturesLayout>(featuresLayout))
        , CommonFloatFeaturesBinarization(std::move(commonFloatFeaturesBinarization))
        , PerFloatFeatureQuantization(std::move(perFloatFeatureQuantization))
        , FloatFeaturesAllowNansInTestOnly(floatFeaturesAllowNansInTestOnly)
        , CatFeaturesPerfectHash(featuresLayout.GetCatFeatureCount())
        , TextDigitizers()
    {
        FeaturesLayout->IgnoreExternalFeatures(ignoredFeatures);
        RuntimeTextProcessingOptions = NCatboostOptions::TRuntimeTextOptions(GetAvailableTextFeatureIndices(*FeaturesLayout),
                                                                             textFeaturesProcessing);
        EmbeddingEstimatorsOptions = NCatboostOptions::TRuntimeEmbeddingOptions(GetAvailableEmbeddingFeatureIndices(*FeaturesLayout),
                                                                                embeddingProcessingOptions);
    }

    void TQuantizedFeaturesInfo::Init(TFeaturesLayout* featuresLayout) {
        FeaturesLayout = MakeIntrusive<TFeaturesLayout>(std::move(*featuresLayout));
        TCatFeaturesPerfectHash catFeaturesPerfectHash(FeaturesLayout->GetCatFeatureCount());
        DoSwap(CatFeaturesPerfectHash, catFeaturesPerfectHash);
    }


    bool TQuantizedFeaturesInfo::EqualTo(const TQuantizedFeaturesInfo& rhs, bool ignoreSparsity) const {
        return EqualWithoutOptionsTo(rhs, ignoreSparsity) &&
            (CommonFloatFeaturesBinarization == rhs.CommonFloatFeaturesBinarization) &&
            (PerFloatFeatureQuantization == rhs.PerFloatFeatureQuantization);
    }

    bool TQuantizedFeaturesInfo::EqualWithoutOptionsTo(
        const TQuantizedFeaturesInfo& rhs,
        bool ignoreSparsity
    ) const {
        return FeaturesLayout->EqualTo(*rhs.FeaturesLayout, ignoreSparsity) &&
            ApproximatelyEqualQuantization(Quantization, rhs.Quantization) &&
            EqualNanModesWithDefault(NanModes, rhs.NanModes) &&
            (CatFeaturesPerfectHash == rhs.CatFeaturesPerfectHash);
    }

    int TQuantizedFeaturesInfo::operator&(IBinSaver& binSaver) {
        AddWithShared(&binSaver, &FeaturesLayout);
        binSaver.AddMulti(
            CommonFloatFeaturesBinarization,
            PerFloatFeatureQuantization,
            FloatFeaturesAllowNansInTestOnly,
            Quantization,
            NanModes,
            CatFeaturesPerfectHash
        );

        return 0;
    }

    ENanMode TQuantizedFeaturesInfo::ComputeNanMode(const TFloatValuesHolder& feature) const {
        auto& floatFeaturesBinarization = GetFloatFeatureBinarization(feature.GetId());
        if (floatFeaturesBinarization.NanMode == ENanMode::Forbidden) {
            return ENanMode::Forbidden;
        }

        bool hasNans;

        if (const auto* denseData = dynamic_cast<const TFloatArrayValuesHolder*>(&feature)) {
            hasNans
                = denseData->GetData()->Find([] (size_t /*idx*/, float value) { return std::isnan(value); });
        } else if (const auto* sparseData = dynamic_cast<const TFloatSparseValuesHolder*>(&feature)) {
            const TConstPolymorphicValuesSparseArray<float, ui32>& sparseArray = sparseData->GetData();
            if (std::isnan(sparseArray.GetDefaultValue())) {
                hasNans = true;
            } else {
                hasNans = false;
                auto blockIterator = sparseArray.GetNonDefaultValues().GetImpl().GetBlockIterator();
                while (auto block = blockIterator->Next()) {
                    for (auto element : block) {
                        if (std::isnan(element)) {
                            hasNans = true;
                            break;
                        }
                    }
                    if (hasNans) {
                        break;
                    }
                }
            }
        } else {
            CB_ENSURE_INTERNAL(false, "TQuantizedFeaturesInfo::ComputeNanMode: unsupported column type");
        }

        if (hasNans) {
            return floatFeaturesBinarization.NanMode;
        }
        return ENanMode::Forbidden;
    }

    ENanMode TQuantizedFeaturesInfo::GetOrComputeNanMode(const TFloatValuesHolder& feature)  {
        const auto floatFeatureIdx = GetPerTypeFeatureIdx<EFeatureType::Float>(feature);
        if (!NanModes.contains(*floatFeatureIdx)) {
            NanModes[*floatFeatureIdx] = ComputeNanMode(feature);
        }
        return NanModes.at(*floatFeatureIdx);
    }

    ENanMode TQuantizedFeaturesInfo::GetNanMode(const TFloatFeatureIdx floatFeatureIdx) const  {
        CheckCorrectPerTypeFeatureIdx(floatFeatureIdx);
        ENanMode nanMode = ENanMode::Forbidden;
        if (NanModes.contains(*floatFeatureIdx)) {
            nanMode = NanModes.at(*floatFeatureIdx);
        }
        return nanMode;
    }

    ui32 TQuantizedFeaturesInfo::CalcMaxCategoricalFeaturesUniqueValuesCountOnLearn() const {
        ui32 maxCategoricalFeaturesUniqueValuesCountOnLearn = 0;
        FeaturesLayout->IterateOverAvailableFeatures<EFeatureType::Categorical>(
            [&] (TCatFeatureIdx catFeatureIdx) {
                auto learnOnlyCount = CatFeaturesPerfectHash.GetUniqueValuesCounts(catFeatureIdx).OnLearnOnly;
                if (learnOnlyCount > maxCategoricalFeaturesUniqueValuesCountOnLearn) {
                    maxCategoricalFeaturesUniqueValuesCountOnLearn = learnOnlyCount;
                }
            }
        );
        return maxCategoricalFeaturesUniqueValuesCountOnLearn;
    }

    TPerfectHashedToHashedCatValuesMap TQuantizedFeaturesInfo::CalcPerfectHashedToHashedCatValuesMap(
        NPar::ILocalExecutor* localExecutor
    ) const {
        // load once and then work with all features in parallel
        LoadCatFeaturePerfectHashToRam();

        const auto& featuresLayout = *GetFeaturesLayout();
        TPerfectHashedToHashedCatValuesMap result(featuresLayout.GetCatFeatureCount());

        localExecutor->ExecRangeWithThrow(
            [&] (int catFeatureIdx) {
                if (!featuresLayout.GetInternalFeatureMetaInfo(
                        (ui32)catFeatureIdx,
                        EFeatureType::Categorical
                    ).IsAvailable)
                {
                    return;
                }

                const auto& catFeaturePerfectHash = GetCategoricalFeaturesPerfectHash(
                    TCatFeatureIdx((ui32)catFeatureIdx)
                );
                auto& perFeatureResult = result[catFeatureIdx];
                perFeatureResult.yresize(catFeaturePerfectHash.GetSize());
                if (catFeaturePerfectHash.DefaultMap) {
                    perFeatureResult[catFeaturePerfectHash.DefaultMap->DstValueWithCount.Value]
                        = catFeaturePerfectHash.DefaultMap->SrcValue;
                }
                for (const auto [hashedCatValue, perfectHash] : catFeaturePerfectHash.Map) {
                    perFeatureResult[perfectHash.Value] = hashedCatValue;
                }
            },
            0,
            SafeIntegerCast<int>(featuresLayout.GetCatFeatureCount()),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );

        return result;
    }


    static ui32 UpdateCheckSumImpl(ui32 init, const TQuantizationWithSerialization& data) {
        return UpdateCheckSum(init, data.Borders, data.DefaultQuantizedBin);
    }

    ui32 TQuantizedFeaturesInfo::CalcCheckSum() const {
        ui32 checkSum = 0;
        auto updateCheskSum = [&checkSum] (const NCatboostOptions::TBinarizationOptions& binarizationOptions) {
            checkSum = UpdateCheckSum(checkSum, binarizationOptions.BorderSelectionType.Get());
            checkSum = UpdateCheckSum(checkSum, binarizationOptions.BorderCount.Get());
            checkSum = UpdateCheckSum(checkSum, binarizationOptions.NanMode.Get());
        };
        updateCheskSum(CommonFloatFeaturesBinarization);
        for (const auto& [featureId, binarizationOptions] : PerFloatFeatureQuantization) {
            checkSum = UpdateCheckSum(checkSum, featureId);
            updateCheskSum(binarizationOptions);
        }
        checkSum = UpdateCheckSum(checkSum, FloatFeaturesAllowNansInTestOnly);
        checkSum = UpdateCheckSum(checkSum, Quantization);
        checkSum = UpdateCheckSum(checkSum, NanModes);
        return checkSum ^ CatFeaturesPerfectHash.CalcCheckSum();
    }


    TPoolQuantizationSchema GetPoolQuantizationSchema(
        const TQuantizedFeaturesInfo& quantizedFeaturesInfo,
        const TVector<NJson::TJsonValue>& classLabels
    ) {
        const auto featuresLayout = quantizedFeaturesInfo.GetFeaturesLayout();
        TVector<TVector<float>> borders;
        TVector<ENanMode> nanModes;
        TVector<size_t> floatFeatureIndices;
        TVector<size_t> catFeatureIndices;
        TVector<TMap<ui32, TValueWithCount>> catFeaturesPerfectHash;

        for (auto flatFeatureIdx : xrange(featuresLayout->GetExternalFeatureCount())) {
            const auto featureMetaInfo = featuresLayout->GetExternalFeatureMetaInfo(flatFeatureIdx);
            if (featureMetaInfo.Type == EFeatureType::Float) {
                const auto floatFeatureIdx = featuresLayout->GetInternalFeatureIdx<EFeatureType::Float>(
                    flatFeatureIdx);
                const auto featureBorders =
                    quantizedFeaturesInfo.HasBorders(floatFeatureIdx) ?
                        quantizedFeaturesInfo.GetBorders(floatFeatureIdx) : TVector<float>();
                const auto featureNanMode =
                    quantizedFeaturesInfo.HasNanMode(floatFeatureIdx) ?
                        quantizedFeaturesInfo.GetNanMode(floatFeatureIdx) : ENanMode::Forbidden;
                borders.push_back(featureBorders);
                nanModes.push_back(featureNanMode);
                floatFeatureIndices.push_back(flatFeatureIdx);
            } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                catFeatureIndices.push_back(flatFeatureIdx);
                const auto catFeatureIdx = featuresLayout->GetInternalFeatureIdx<EFeatureType::Categorical>(
                    flatFeatureIdx
                );
                catFeaturesPerfectHash.push_back(
                    quantizedFeaturesInfo.GetCategoricalFeaturesPerfectHash(catFeatureIdx).ToMap()
                );
            } else {
                CB_ENSURE_INTERNAL(
                    false,
                    "building quantization results is supported only for numerical and categorical features"
                );
            }
        }
        return TPoolQuantizationSchema {
            std::move(floatFeatureIndices),
            std::move(borders),
            std::move(nanModes),
            classLabels,
            std::move(catFeatureIndices),
            std::move(catFeaturesPerfectHash)
        };
    }

}
