#include "dataset_rows_reader.h"

#include "meta_info.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/data/cb_dsv_loader.h>
#include <catboost/libs/data/libsvm_loader.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/plain_options_helper.h>

#include <library/cpp/json/json_reader.h>

#include <util/generic/cast.h>
#include <util/system/compiler.h>

using namespace NCB;


static void ParseOptions(
    const TString& plainJsonParamsAsString,
    TDsvFormatOptions* dsvFormatOptions,
    TVector<ui32>* ignoredFeatures
) {
    NJson::TJsonValue plainJsonParams;
    try {
        NJson::ReadJsonTree(plainJsonParamsAsString, &plainJsonParams, /*throwOnError*/ true);
        if (plainJsonParams.Has("has_header")) {
            // it will be passed to TRawDatasetRowsReader directly because it will be different
            // for different splits
            plainJsonParams.EraseValue("has_header");
        }
        if (plainJsonParams.Has("delimiter")) {
            const NJson::TJsonValue& delimiterJson = plainJsonParams["delimiter"];
            CB_ENSURE(delimiterJson.IsString(), "delimiter value is not string");
            const TString& delimiterString = delimiterJson.GetString();
            CB_ENSURE(delimiterString.size() == 1, "delimiter must be single char");
            dsvFormatOptions->Delimiter = delimiterString[0];
            plainJsonParams.EraseValue("delimiter");
        }
    } catch (const std::exception& e) {
        throw TCatBoostException() << "Error while parsing data loading params JSON: " << e.what();
    }
    NJson::TJsonValue jsonParams;
    NJson::TJsonValue outputJsonParams;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &jsonParams, &outputJsonParams);
    NCatboostOptions::TCatBoostOptions catBoostOptions(NCatboostOptions::LoadOptions(jsonParams));
    *ignoredFeatures = catBoostOptions.DataProcessingOptions.Get().IgnoredFeatures.Get();
}


class TRawDatasetRowsReaderVisitor final : public IRawObjectsOrderDataVisitor {
public:
    TRawDatasetRowsReaderVisitor(TRawDatasetRowsReader* parent, NPar::ILocalExecutor* localExecutor)
        : Parent(parent)
        , LocalExecutor(localExecutor)
    {}

    i32 GetBlockSize() const {
        return SafeIntegerCast<i32>(Rows.size());
    }

    const TRawDatasetRow& GetRow(i32 objectIdx) {
        return Rows[objectIdx];
    }

    // separate method because they can be loaded from a separate data source
     void SetGroupWeights(TVector<float>&& groupWeights) override {
         Y_UNUSED(groupWeights);
         CB_ENSURE(false, "SetGroupWeights is incompatible with blocked processing");
     }

     // separate method because they can be loaded from a separate data source
     void SetBaseline(TVector<TVector<float>>&& baseline) override {
         Y_UNUSED(baseline);
         CB_ENSURE(false, "SetBaseline is incompatible with blocked processing");
     }

     void SetPairs(TRawPairsData&& pairs) override {
         Y_UNUSED(pairs);
         CB_ENSURE(false, "SetPairs is incompatible with blocked processing");
     }

     void SetTimestamps(TVector<ui64>&& timestamps) override {
         Y_UNUSED(timestamps);
         CB_ENSURE(false, "SetTimestamps is incompatible with blocked processing");
     }

     TMaybeData<TConstArrayRef<TGroupId>> GetGroupIds() const override {
         CB_ENSURE(false, "GetGroupIds is incompatible with blocked processing");
         return TMaybeData<TConstArrayRef<TGroupId>>();
     }

    void Start(
        bool inBlock, // subset processing - Start/Finish is called for each block
        const TDataMetaInfo& metaInfo,
        bool haveUnknownNumberOfSparseFeatures,
        ui32 objectCount,
        EObjectsOrder objectsOrder,

        // keep necessary resources for data to be available (memory mapping for a file for example)
        TVector<TIntrusivePtr<IResourceHolder>> resourceHolders
    ) override {
        Y_UNUSED(inBlock);
        Y_UNUSED(objectsOrder);
        Y_UNUSED(resourceHolders);

        if (FirstBlock) {
            TIntermediateDataMetaInfo intermediateMetaInfo(metaInfo, haveUnknownNumberOfSparseFeatures);
            IsSparse = intermediateMetaInfo.HasSparseFeatures();
            FeatureCount = metaInfo.GetFeatureCount();

            CB_ENSURE(metaInfo.TargetCount <= 1, "Multiple targets are not supported yet");

            Parent->SetMetaInfo(std::move(intermediateMetaInfo));
        }
        Rows.resize(objectCount);
        BaselinesStorage.yresize(objectCount * metaInfo.BaselineCount);

        if (IsSparse) {
            if (!FirstBlock) {
                LocalExecutor->ExecRangeBlockedWithThrow(
                    [&] (int i) {
                        Rows[i].SparseFloatFeaturesIndices.clear(),
                        Rows[i].SparseFloatFeaturesValues.clear();
                    },
                    0,
                    SafeIntegerCast<int>(objectCount),
                    /*batchSizeOrZeroForAutoBatchSize*/ 0,
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );
            }
        } else {
            DenseFloatFeaturesStorage.yresize(objectCount * FeatureCount);
            ui32 baselineCount = metaInfo.BaselineCount;
            BaselinesStorage.yresize(objectCount * baselineCount);
            if (FirstBlock) {
                double* featuresDataBegin = DenseFloatFeaturesStorage.data();
                double* baselineDataBegin = BaselinesStorage.data();
                LocalExecutor->ExecRangeBlockedWithThrow(
                    [&, featuresDataBegin, baselineDataBegin, baselineCount, featureCount=FeatureCount] (int i) {
                        Rows[i].DenseFloatFeatures = TArrayRef<double>(
                            featuresDataBegin + ((ui32)i * featureCount),
                            featureCount
                        );
                        if (baselineCount) {
                            Rows[i].Baselines = TArrayRef<double>(
                                baselineDataBegin + ((ui32)i * baselineCount),
                                baselineCount
                            );
                        }
                    },
                    0,
                    SafeIntegerCast<int>(objectCount),
                    /*batchSizeOrZeroForAutoBatchSize*/ 0,
                    NPar::TLocalExecutor::WAIT_COMPLETE
                );
            }
        }
        FirstBlock = false;
    }

    void StartNextBlock(ui32 blockSize) override {
        Y_UNUSED(blockSize);
    }

    // TCommonObjectsData
    void AddGroupId(ui32 localObjectIdx, TGroupId value) override {
        Rows[localObjectIdx].GroupId = reinterpret_cast<const i64&>(value);
    }
    void AddSubgroupId(ui32 localObjectIdx, TSubgroupId value) override {
        Rows[localObjectIdx].SubgroupId = reinterpret_cast<const i32&>(value);
    }
    void AddGroupId(ui32 /*localObjectIdx*/, const TString& /*value*/) override {
        CB_ENSURE_INTERNAL(false, "unsupported function");
    }
    void AddSubgroupId(ui32 /*localObjectIdx*/, const TString& /*value*/) override {
        CB_ENSURE_INTERNAL(false, "unsupported function");
    }
    void AddSampleId(ui32 /*localObjectIdx*/, const TString& /*value*/) override {
        CB_ENSURE_INTERNAL(false, "unsupported function");
    }
    void AddTimestamp(ui32 localObjectIdx, ui64 value) override {
        Rows[localObjectIdx].Timestamp = SafeIntegerCast<i64>(value);
    }

    // TRawObjectsData
    void AddFloatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, float feature) override {
        if (IsSparse) {
            Rows[localObjectIdx].SparseFloatFeaturesIndices.push_back(SafeIntegerCast<i32>(flatFeatureIdx));
            Rows[localObjectIdx].SparseFloatFeaturesValues.push_back(feature);
        } else {
            DenseFloatFeaturesStorage[localObjectIdx*FeatureCount + flatFeatureIdx] = feature;
        }
    }
    void AddAllFloatFeatures(ui32 localObjectIdx, TConstArrayRef<float> features) override {
        Y_ASSERT(!IsSparse);
        Y_ASSERT(features.size() == FeatureCount);
        Copy(
            features.begin(),
            features.end(),
            DenseFloatFeaturesStorage.data() + localObjectIdx * FeatureCount
        );
    }
    void AddAllFloatFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<float, ui32> features
    ) override {
        Y_ASSERT(IsSparse);
        auto& row = Rows[localObjectIdx];
        features.ForEachNonDefault(
            [&] (ui32 perTypeFeatureIdx, float value) {
                row.SparseFloatFeaturesIndices.push_back(perTypeFeatureIdx);
                row.SparseFloatFeaturesValues.push_back(value);
            }
        );
    }

    // for sparse float features default value is always assumed to be 0.0f

    ui32 GetCatFeatureValue(ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(flatFeatureIdx);
        Y_UNUSED(feature);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }
    // localObjectIdx may be used as hint for sampling
    ui32 GetCatFeatureValue(ui32 /* localObjectIdx */, ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(flatFeatureIdx);
        Y_UNUSED(feature);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }
    void AddCatFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(flatFeatureIdx);
        Y_UNUSED(feature);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }
    void AddAllCatFeatures(ui32 localObjectIdx, TConstArrayRef<ui32> features) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(features);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }
    void AddAllCatFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<ui32, ui32> features
    ) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(features);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }

    // for sparse data
    void AddCatFeatureDefaultValue(ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(flatFeatureIdx);
        Y_UNUSED(feature);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }

    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, TStringBuf feature) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(flatFeatureIdx);
        Y_UNUSED(feature);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }
    void AddTextFeature(ui32 localObjectIdx, ui32 flatFeatureIdx, const TString& feature) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(flatFeatureIdx);
        Y_UNUSED(feature);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }
    void AddAllTextFeatures(ui32 localObjectIdx, TConstArrayRef<TString> features) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(features);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }
    void AddAllTextFeatures(
        ui32 localObjectIdx,
        TConstPolymorphicValuesSparseArray<TString, ui32> features
    ) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(features);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }

    void AddEmbeddingFeature(
        ui32 localObjectIdx,
        ui32 flatFeatureIdx,
        TMaybeOwningConstArrayHolder<float> feature
    ) override {
        Y_UNUSED(localObjectIdx);
        Y_UNUSED(flatFeatureIdx);
        Y_UNUSED(feature);
        CB_ENSURE(false, "Non-float features are not supported yet");
    }

    // TRawTargetData

    void AddTarget(ui32 localObjectIdx, const TString& value) override {
        Rows[localObjectIdx].StringTarget = value;
    }
    void AddTarget(ui32 localObjectIdx, float value) override {
        Rows[localObjectIdx].FloatTarget = value;
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, const TString& value) override {
        CB_ENSURE(flatTargetIdx == 0, "Multiple targets are not supported yet");
        AddTarget(localObjectIdx, value);
    }
    void AddTarget(ui32 flatTargetIdx, ui32 localObjectIdx, float value) override {
        CB_ENSURE(flatTargetIdx == 0, "Multiple targets are not supported yet");
        AddTarget(localObjectIdx, value);
    }
    void AddBaseline(ui32 localObjectIdx, ui32 baselineIdx, float value) override {
        Rows[localObjectIdx].Baselines[baselineIdx] = value;
    }
    void AddWeight(ui32 localObjectIdx, float value) override {
        Rows[localObjectIdx].Weight = value;
    }
    void AddGroupWeight(ui32 localObjectIdx, float value) override {
        Rows[localObjectIdx].GroupWeight = value;
    }

    void Finish() override {}

private:
    TRawDatasetRowsReader* Parent;
    NPar::ILocalExecutor* LocalExecutor;

    bool FirstBlock = true;
    bool IsSparse = false;
    ui32 FeatureCount = 0; // when it's known

    TVector<TRawDatasetRow> Rows;
    TVector<double> DenseFloatFeaturesStorage; // [objectCount x ]
    TVector<double> BaselinesStorage; // [objectCount x BaselineCount]
};



TRawDatasetRowsReader::TRawDatasetRowsReader(
    const TString& schema,
    ILineDataReader* lineReader,
    const TString& columnDescriptionPathWithScheme,
    const TVector<TColumn>& columnsDescription,
    const TString& plainJsonParamsAsString,
    bool hasHeader,
    i32 blockSize,
    i32 threadCount
) {
    THolder<ILineDataReader> lineReaderHolder(lineReader);

    CB_ENSURE(threadCount >= 1, "threadCount must be >= 1");

    TDsvFormatOptions dsvFormatOptions;
    dsvFormatOptions.HasHeader = hasHeader;
    TVector<ui32> ignoredFeatures;
    ParseOptions(plainJsonParamsAsString, &dsvFormatOptions, &ignoredFeatures);

    LocalExecutor.RunAdditionalThreads(threadCount - 1);

    THolder<ICdProvider> cdProvider;
    if (!columnDescriptionPathWithScheme.empty()) {
        cdProvider = MakeCdProviderFromFile(TPathWithScheme(columnDescriptionPathWithScheme));
    } else {
        cdProvider = MakeCdProviderFromArray(columnsDescription);
    }

    TLineDataLoaderPushArgs args{
        std::move(lineReaderHolder),

        TDatasetLoaderCommonArgs {
            /*pairsFilePath*/ TPathWithScheme(),
            /*groupWeightsFilePath*/ TPathWithScheme(),
            /*baselineFilePath*/ TPathWithScheme(),
            /*timestampsFilePath*/ TPathWithScheme(),
            /*featureNamesPath*/ TPathWithScheme(),
            /*poolMetaInfoPath*/ TPathWithScheme(),
            /*classLabels*/ TVector<NJson::TJsonValue>(),
            dsvFormatOptions,
            std::move(cdProvider),
            ignoredFeatures,
            EObjectsOrder::Undefined,
            SafeIntegerCast<ui32>(blockSize),
            /*loadSubset*/ TDatasetSubset(),
            /*LoadColumnsAsString*/ false,
            /*LoadSampleIds*/ false,
            /*ForceUnitAutoPairWeights*/ false,
            &LocalExecutor
        }
    };

    if (schema == "dsv") {
        Loader.Reset(new TCBDsvDataLoader(std::move(args)));
    } else if (schema == "libsvm") {
        Loader.Reset(new TLibSvmDataLoader(std::move(args)));
    } else {
        ythrow TCatBoostException() << "Unsupported schema: \"" << schema << "\"";
    }

    THolder<TRawDatasetRowsReaderVisitor> visitorHolder(
        new TRawDatasetRowsReaderVisitor(this, &LocalExecutor)
    );
    Visitor = visitorHolder.Get();
    VisitorHolder = std::move(visitorHolder);
}

i32 TRawDatasetRowsReader::ReadNextBlock() {
    if (Loader->DoBlock(Visitor)) {
        return Visitor->GetBlockSize();
    } else {
        return 0;
    }
}

const TRawDatasetRow& TRawDatasetRowsReader::GetRow(i32 objectIdx) {
    return Visitor->GetRow(objectIdx);
}
