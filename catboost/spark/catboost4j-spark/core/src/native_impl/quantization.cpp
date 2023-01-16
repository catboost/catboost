#include "quantization.h"

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/meta_info.h>
#include <catboost/libs/data/quantization.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>

#include <library/cpp/dbg_output/dump.h>
#include <library/cpp/grid_creator/binarization.h>
#include <library/cpp/json/json_reader.h>
#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>
#include <util/system/rwlock.h>

#include <limits>


using namespace NCB;


TQuantizedFeaturesInfoPtr PrepareQuantizationParameters(
    const TFeaturesLayout& featuresLayout,
    const TString& plainJsonParamsAsString
) {
    NJson::TJsonValue plainJsonParams;

    TMaybe<TString> inputBorders;
    try {
        NJson::ReadJsonTree(plainJsonParamsAsString, &plainJsonParams, /*throwOnError*/ true);

        if (plainJsonParams.Has("input_borders")) {
            const NJson::TJsonValue& inputBordersJson = plainJsonParams["input_borders"];
            CB_ENSURE(inputBordersJson.IsString(), "input_borders value is not a string");
            inputBorders = inputBordersJson.GetString();
            plainJsonParams.EraseValue("input_borders");
        }
    } catch (const std::exception& e) {
        throw TCatBoostException() << "Error while parsing quantization params JSON: " << e.what();
    }

    TDataMetaInfo metaInfo;
    metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(featuresLayout);

    TQuantizationOptions quantizationOptions;
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo;

    PrepareQuantizationParameters(
        plainJsonParams,
        metaInfo,
        inputBorders,
        &quantizationOptions,
        &quantizedFeaturesInfo
    );

    return quantizedFeaturesInfo;
}

TNanModeAndBordersBuilder::TNanModeAndBordersBuilder(
    TQuantizedFeaturesInfoPtr quantizedFeaturesInfo
)
    : QuantizedFeaturesInfo(quantizedFeaturesInfo)
{
    const TFeaturesLayout& featuresLayout = *(QuantizedFeaturesInfo->GetFeaturesLayout());

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx idx) {
            if (!QuantizedFeaturesInfo->HasBorders(idx)) {
                FeatureIndicesToCalc.push_back(
                    featuresLayout.GetExternalFeatureIdx(*idx, EFeatureType::Float)
                );
            }
        }
    );

    Data.resize(FeatureIndicesToCalc.size());
}

void TNanModeAndBordersBuilder::SetSampleSize(i32 sampleSize) {
    SampleSize = sampleSize;
    for (auto& featureData : Data) {
        featureData.reserve(sampleSize);
    }
}

void TNanModeAndBordersBuilder::AddSample(TConstArrayRef<double> objectData) {
    for (auto i : xrange(FeatureIndicesToCalc.size())) {
        double value = objectData[FeatureIndicesToCalc[i]];
        if (!IsNan(value)) {
            Data[i].push_back((float)value);
        }
    }
}

void TNanModeAndBordersBuilder::CalcBordersWithoutNans(i32 threadCount) {
    CB_ENSURE(threadCount >= 1);

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    const TFeaturesLayout& featuresLayout = *(QuantizedFeaturesInfo->GetFeaturesLayout());

    HasNans.yresize(FeatureIndicesToCalc.size());
    QuantizationWithoutNans.resize(FeatureIndicesToCalc.size());
    localExecutor.ExecRangeWithThrow(
        [&] (int i) {
            HasNans[i] = Data[i].size() < SampleSize;

            auto floatFeatureIdx =
                featuresLayout.GetInternalFeatureIdx<EFeatureType::Float>(FeatureIndicesToCalc[i]);

            if (Data[i].empty()) {
                CATBOOST_DEBUG_LOG << "Float Feature #" << *floatFeatureIdx
                    << ": sample data contains only NaNs" << Endl;
            } else {
                const NCatboostOptions::TBinarizationOptions& binarizationOptions =
                    QuantizedFeaturesInfo->GetFloatFeatureBinarization(*floatFeatureIdx);

                QuantizationWithoutNans[i] = NSplitSelection::BestSplit(
                    NSplitSelection::TFeatureValues(std::move(Data[i])),
                    /*featureValuesMayContainNans*/ false,
                    binarizationOptions.BorderCount.Get(),
                    binarizationOptions.BorderSelectionType.Get(),
                    /*quantizedDefaultBinFraction*/ Nothing(),
                    /*initialBorders*/ Nothing()
                );
            }
        },
        0,
        SafeIntegerCast<int>(FeatureIndicesToCalc.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
}

void TNanModeAndBordersBuilder::Finish(TConstArrayRef<i8> hasNans) {
    TFeaturesLayout& featuresLayout = *(QuantizedFeaturesInfo->GetFeaturesLayout());

    for (auto i : xrange(FeatureIndicesToCalc.size())) {
        auto flatFeatureIdx = FeatureIndicesToCalc[i];
        if (!hasNans.empty()) {
            HasNans[i] = hasNans[flatFeatureIdx] == 1;
        }

        auto floatFeatureIdx = featuresLayout.GetInternalFeatureIdx<EFeatureType::Float>(flatFeatureIdx);

        const NCatboostOptions::TBinarizationOptions& binarizationOptions =
            QuantizedFeaturesInfo->GetFloatFeatureBinarization(*floatFeatureIdx);

        ENanMode nanMode;
        if (HasNans[i]) {
            CB_ENSURE(
                binarizationOptions.NanMode != ENanMode::Forbidden,
                "Feature #" << flatFeatureIdx << ": There are nan factors and nan values for "
                " float features are not allowed. Set nan_mode != Forbidden."
            );
            nanMode = binarizationOptions.NanMode;
        } else {
            nanMode = ENanMode::Forbidden;
        }

        NSplitSelection::TQuantization quantization = std::move(QuantizationWithoutNans[i]);
        if (quantization.Borders.empty()) {
            CATBOOST_DEBUG_LOG << "Float Feature #" << *floatFeatureIdx << " is empty" << Endl;

            featuresLayout.IgnoreExternalFeature(flatFeatureIdx);
        } else {
            if (nanMode == ENanMode::Min) {
                quantization.Borders.insert(quantization.Borders.begin(), std::numeric_limits<float>::lowest());
            } else if (nanMode == ENanMode::Max) {
                quantization.Borders.push_back(std::numeric_limits<float>::max());
            }
            QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanMode);
            QuantizedFeaturesInfo->SetQuantization(floatFeatureIdx, std::move(quantization));
        }
    }
}

TQuantizedObjectsDataProviderPtr Quantize(
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    NCB::TRawObjectsDataProviderPtr* rawObjectsDataProvider, // moved into
    NPar::TLocalExecutor* localExecutor
) {
    TQuantizationOptions options;
    options.BundleExclusiveFeatures = false;
    options.PackBinaryFeaturesForCpu = false;
    options.GroupFeaturesForCpu = false;

    TRestorableFastRng64 rand(0);

    return Quantize(
        options,
        std::move(*rawObjectsDataProvider),
        quantizedFeaturesInfo,
        &rand,
        localExecutor
    );
}

void GetActiveFeaturesIndices(
    NCB::TFeaturesLayoutPtr featuresLayout,
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TVector<i32>* ui8FlatIndices,
    TVector<i32>* ui16FlatIndices,
    TVector<i32>* ui32FlatIndices
) {
    ui8FlatIndices->clear();
    ui16FlatIndices->clear();
    ui32FlatIndices->clear();

    featuresLayout->IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx idx) {
            i32 flatIdx = SafeIntegerCast<i32>(
                featuresLayout->GetExternalFeatureIdx(*idx, EFeatureType::Float)
            );
            if (quantizedFeaturesInfo->GetBorders(idx).size() > 255) {
                ui16FlatIndices->push_back(flatIdx);
            } else {
                ui8FlatIndices->push_back(flatIdx);
            }
        }
    );
    featuresLayout->IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx idx) {
            i32 flatIdx = SafeIntegerCast<i32>(
                featuresLayout->GetExternalFeatureIdx(*idx, EFeatureType::Categorical)
            );

            ui32 uniqValuesCount = quantizedFeaturesInfo->GetUniqueValuesCounts(idx).OnAll;
            if (uniqValuesCount > ((ui32)Max<ui16>() + 1)) {
                ui32FlatIndices->push_back(flatIdx);
            } else if (uniqValuesCount > ((ui32)Max<ui8>() + 1)) {
                ui16FlatIndices->push_back(flatIdx);
            } else {
                ui8FlatIndices->push_back(flatIdx);
            }
        }
    );
}
