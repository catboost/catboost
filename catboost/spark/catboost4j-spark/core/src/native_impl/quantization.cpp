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


static TQuantizedFeaturesInfoPtr PrepareQuantizationParameters(
    const TString& plainJsonParamsAsString,
    int featureCount,
    const TVector<TString>& featureNames
) throw (yexception) {
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
    metaInfo.FeaturesLayout = MakeIntrusive<TFeaturesLayout>(
        SafeIntegerCast<ui32>(featureCount),
        /*catFeatureIndices*/ TVector<ui32>{}, // TODO(akhropov): cat features are not supported yet
        featureNames
    );

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
    const TString& plainJsonParamsAsString,
    i32 featureCount,
    const TVector<TString>& featureNames,
    i32 sampleSize
) throw (yexception)
    : SampleSize(SafeIntegerCast<size_t>(sampleSize))
{
    QuantizedFeaturesInfo = PrepareQuantizationParameters(
        plainJsonParamsAsString,
        featureCount,
        featureNames
    );

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
    for (auto& featureData : Data) {
        featureData.reserve(sampleSize);
    }
}

void TNanModeAndBordersBuilder::AddSample(TConstArrayRef<double> objectData) throw (yexception) {
    for (auto i : xrange(FeatureIndicesToCalc.size())) {
        double value = objectData[FeatureIndicesToCalc[i]];
        if (!IsNan(value)) {
            Data[i].push_back((float)value);
        }
    }
}

TQuantizedFeaturesInfoPtr TNanModeAndBordersBuilder::Finish(i32 threadCount) throw (yexception) {
    CB_ENSURE(threadCount >= 1);

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    TFeaturesLayout& featuresLayout = *(QuantizedFeaturesInfo->GetFeaturesLayout());

    localExecutor.ExecRangeWithThrow(
        [&] (int i) {
            auto floatFeatureIdx =
                featuresLayout.GetInternalFeatureIdx<EFeatureType::Float>(FeatureIndicesToCalc[i]);

            const NCatboostOptions::TBinarizationOptions& binarizationOptions =
                QuantizedFeaturesInfo->GetFloatFeatureBinarization(*floatFeatureIdx);

            ENanMode nanMode;
            if (Data[i].size() < SampleSize) { // has NaNs
                CB_ENSURE(
                    binarizationOptions.NanMode != ENanMode::Forbidden,
                    "Feature #" << FeatureIndicesToCalc[i] << ": There are nan factors and nan values for "
                    " float features are not allowed. Set nan_mode != Forbidden."
                );
                nanMode = binarizationOptions.NanMode;
            } else {
                nanMode = ENanMode::Forbidden;
            }

            NSplitSelection::TQuantization quantization;
            if (Data[i].empty()) {
                CATBOOST_DEBUG_LOG << "Float Feature #" << *floatFeatureIdx
                    << ": sample data contains only NaNs" << Endl;
            } else {
                quantization = NSplitSelection::BestSplit(
                    NSplitSelection::TFeatureValues(std::move(Data[i])),
                    /*featureValuesMayContainNans*/ false,
                    binarizationOptions.BorderCount.Get(),
                    binarizationOptions.BorderSelectionType.Get(),
                    /*quantizedDefaultBinFraction*/ Nothing(),
                    /*initialBorders*/ Nothing()
                );
                if (nanMode == ENanMode::Min) {
                    quantization.Borders.insert(quantization.Borders.begin(), std::numeric_limits<float>::lowest());
                } else if (nanMode == ENanMode::Max) {
                    quantization.Borders.push_back(std::numeric_limits<float>::max());
                }
            }
            {
                TWriteGuard writeGuard(QuantizedFeaturesInfo->GetRWMutex());
                if (quantization.Borders.empty()) {
                    CATBOOST_DEBUG_LOG << "Float Feature #" << *floatFeatureIdx << " is empty" << Endl;

                    featuresLayout.IgnoreExternalFeature(FeatureIndicesToCalc[i]);
                }
                QuantizedFeaturesInfo->SetNanMode(floatFeatureIdx, nanMode);
                QuantizedFeaturesInfo->SetQuantization(floatFeatureIdx, std::move(quantization));
            }
        },
        0,
        SafeIntegerCast<int>(FeatureIndicesToCalc.size()),
        NPar::TLocalExecutor::WAIT_COMPLETE
    );
    return QuantizedFeaturesInfo;
}

TQuantizedObjectsDataProviderPtr Quantize(
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    NCB::TRawObjectsDataProviderPtr* rawObjectsDataProvider, // moved into
    int threadCount
) throw (yexception) {
    TQuantizationOptions options;
    options.BundleExclusiveFeatures = false;
    options.PackBinaryFeaturesForCpu = false;
    options.GroupFeaturesForCpu = false;

    TRestorableFastRng64 rand(0);

    CB_ENSURE(threadCount >= 1);

    NPar::TLocalExecutor localExecutor;
    localExecutor.RunAdditionalThreads(threadCount - 1);

    return Quantize(
        options,
        std::move(*rawObjectsDataProvider),
        quantizedFeaturesInfo,
        &rand,
        &localExecutor
    );
}

void GetActiveFloatFeaturesIndices(
    NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
    TVector<i32>* ui8Indices,
    TVector<i32>* ui16Indices
) throw (yexception) {
    const auto& featuresLayout = *(quantizedFeaturesInfo->GetFeaturesLayout());

    ui8Indices->clear();
    ui16Indices->clear();

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx idx) {
            if (quantizedFeaturesInfo->GetBorders(idx).size() > 255) {
                ui16Indices->push_back(SafeIntegerCast<i32>(*idx));
            } else {
                ui8Indices->push_back(SafeIntegerCast<i32>(*idx));
            }
        }
    );
}
