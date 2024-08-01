#include "baseline.h"
#include "load_data.h"

#include "cb_dsv_loader.h"
#include "data_provider_builders.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/data_util/exists_checker.h>

#include <util/datetime/base.h>


namespace NCB {

    TDataProviderPtr ReadDataset(
        TMaybe<ETaskType> taskType,
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& graphFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const TPathWithScheme& poolMetaInfoPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TDatasetSubset loadSubset,
        bool loadSampleIds,
        bool forceUnitAutoPairWeights,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::ILocalExecutor* localExecutor
    ) {
        CB_ENSURE_INTERNAL(!baselineFilePath.Inited() || classLabels, "ClassLabels must be specified if baseline file is specified");
        if (classLabels) {
            UpdateClassLabelsFromBaselineFile(baselineFilePath, *classLabels);
        }
        auto datasetLoader = GetProcessor<IDatasetLoader>(
            poolPath, // for choosing processor

            // processor args
            TDatasetLoaderPullArgs {
                poolPath,

                TDatasetLoaderCommonArgs {
                    pairsFilePath,
                    graphFilePath,
                    groupWeightsFilePath,
                    baselineFilePath,
                    timestampsFilePath,
                    featureNamesPath,
                    poolMetaInfoPath,
                    classLabels ? **classLabels : TVector<NJson::TJsonValue>(),
                    columnarPoolFormatParams.DsvFormat,
                    MakeCdProviderFromFile(columnarPoolFormatParams.CdFilePath),
                    ignoredFeatures,
                    objectsOrder,
                    10000, // TODO: make it a named constant
                    loadSubset,
                    /*LoadColumnsAsString*/ loadSampleIds,
                    /*LoadSampleIds*/ loadSampleIds,
                    forceUnitAutoPairWeights,
                    localExecutor
                }
            }
        );

        TDataProviderBuilderOptions builderOptions;
        builderOptions.GpuDistributedFormat = !loadSubset.HasFeatures && taskType && *taskType == ETaskType::GPU
            && EDatasetVisitorType::QuantizedFeatures == datasetLoader->GetVisitorType()
            && poolPath.Inited() && IsSharedFs(poolPath);
        builderOptions.PoolPath = poolPath;

        THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
            datasetLoader->GetVisitorType(),
            builderOptions,
            loadSubset,
            localExecutor
        );
        CB_ENSURE_INTERNAL(
            dataProviderBuilder,
            "Failed to create data provider builder for visitor of type " << datasetLoader->GetVisitorType()
        );

        datasetLoader->DoIfCompatible(dynamic_cast<IDatasetVisitor*>(dataProviderBuilder.Get()));
        return dataProviderBuilder->GetResult();
    }


    TDataProviderPtr ReadDataset(
        TMaybe<ETaskType> taskType,
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& graphFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const TPathWithScheme& poolMetaInfoPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        int threadCount,
        bool verbose,
        bool loadSampleIds,
        bool forceUnitAutoPairWeights,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);

        TSetLoggingVerboseOrSilent inThisScope(verbose);

        TDataProviderPtr dataProviderPtr = ReadDataset(
            taskType,
            poolPath,
            pairsFilePath,
            graphFilePath,
            groupWeightsFilePath,
            timestampsFilePath,
            baselineFilePath,
            featureNamesPath,
            poolMetaInfoPath,
            columnarPoolFormatParams,
            ignoredFeatures,
            objectsOrder,
            TDatasetSubset::MakeColumns(),
            loadSampleIds,
            forceUnitAutoPairWeights,
            classLabels,
            &localExecutor
        );

        return dataProviderPtr;
    }


    TDataProviderPtr ReadDataset(
        THolder<ILineDataReader>&& poolReader,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& graphFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const TPathWithScheme& poolMetaInfoPath, // can be uninited
        const TDsvFormatOptions& poolFormat,
        const TVector<TColumn>& columnsDescription, // TODO(smirnovpavel): TVector<EColumn>
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        bool loadSampleIds,
        bool forceUnitAutoPairWeights,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::ILocalExecutor* localExecutor
    ) {
        const auto loadSubset = TDatasetSubset::MakeColumns();
        THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
            EDatasetVisitorType::RawObjectsOrder,
            TDataProviderBuilderOptions{},
            loadSubset,
            localExecutor
        );
        CB_ENSURE_INTERNAL(
            dataProviderBuilder,
            "Failed to create data provider builder for visitor of type RawObjectsOrder";
        );

        TCBDsvDataLoader datasetLoader(
            TLineDataLoaderPushArgs {
                std::move(poolReader),

                TDatasetLoaderCommonArgs {
                    pairsFilePath,
                    graphFilePath,
                    groupWeightsFilePath,
                    baselineFilePath,
                    timestampsFilePath,
                    featureNamesPath,
                    poolMetaInfoPath,
                    classLabels ? **classLabels : TVector<NJson::TJsonValue>(),
                    poolFormat,
                    MakeCdProviderFromArray(columnsDescription),
                    ignoredFeatures,
                    objectsOrder,
                    10000, // TODO: make it a named constant
                    loadSubset,
                    /*LoadColumnsAsString*/ loadSampleIds,
                    loadSampleIds,
                    forceUnitAutoPairWeights,
                    localExecutor
                }
            }
        );
        datasetLoader.DoIfCompatible(dynamic_cast<IDatasetVisitor*>(dataProviderBuilder.Get()));
        return dataProviderBuilder->GetResult();
    }

    TDataProviders ReadTrainDatasets(
        TMaybe<ETaskType> taskType,
        const NCatboostOptions::TPoolLoadParams& loadOptions,
        EObjectsOrder objectsOrder,
        bool readTestData,
        TDatasetSubset learnDatasetSubset,
        TConstArrayRef<TDatasetSubset> testDatasetSubsets,
        bool forceUnitAutoPairWeights,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::ILocalExecutor* const executor,
        TProfileInfo* const profile
    ) {
        if (readTestData) {
            loadOptions.Validate();
        } else {
            loadOptions.ValidateLearn();
        }

        TDataProviders dataProviders;

        if (loadOptions.LearnSetPath.Inited()) {
            CATBOOST_DEBUG_LOG << "Loading features..." << Endl;
            auto start = Now();
            dataProviders.Learn = ReadDataset(
                taskType,
                loadOptions.LearnSetPath,
                loadOptions.PairsFilePath,
                loadOptions.GraphFilePath,
                loadOptions.GroupWeightsFilePath,
                loadOptions.TimestampsFilePath,
                loadOptions.BaselineFilePath,
                loadOptions.FeatureNamesPath,
                loadOptions.PoolMetaInfoPath,
                loadOptions.ColumnarPoolFormatParams,
                loadOptions.IgnoredFeatures,
                objectsOrder,
                learnDatasetSubset,
                /*loadSampleIds*/ false,
                forceUnitAutoPairWeights,
                classLabels,
                executor
            );
            CATBOOST_DEBUG_LOG << "Loading features time: " << (Now() - start).Seconds() << Endl;
            if (profile) {
                profile->AddOperation("Build learn pool");
            }
        }
        dataProviders.Test.resize(0);

        if (readTestData) {
            CATBOOST_DEBUG_LOG << "Loading test..." << Endl;
            for (int testIdx = 0; testIdx < loadOptions.TestSetPaths.ysize(); ++testIdx) {
                const NCB::TPathWithScheme& testSetPath = loadOptions.TestSetPaths[testIdx];
                const NCB::TPathWithScheme& testPairsFilePath =
                        testIdx == 0 ? loadOptions.TestPairsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testGraphFilePath =
                        testIdx == 0 ? loadOptions.TestGraphFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testGroupWeightsFilePath =
                    testIdx == 0 ? loadOptions.TestGroupWeightsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testTimestampsFilePath =
                    testIdx == 0 ? loadOptions.TestTimestampsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testBaselineFilePath =
                    testIdx == 0 ? loadOptions.TestBaselineFilePath : NCB::TPathWithScheme();

                TDataProviderPtr testDataProvider = ReadDataset(
                    taskType,
                    testSetPath,
                    testPairsFilePath,
                    testGraphFilePath,
                    testGroupWeightsFilePath,
                    testTimestampsFilePath,
                    testBaselineFilePath,
                    loadOptions.FeatureNamesPath,
                    loadOptions.PoolMetaInfoPath,
                    loadOptions.ColumnarPoolFormatParams,
                    loadOptions.IgnoredFeatures,
                    objectsOrder,
                    testDatasetSubsets[testIdx],
                    /*loadSampleIds*/ false,
                    forceUnitAutoPairWeights,
                    classLabels,
                    executor
                );
                dataProviders.Test.push_back(std::move(testDataProvider));
                if (profile && (testIdx + 1 == loadOptions.TestSetPaths.ysize())) {
                    profile->AddOperation("Build test pool");
                }
            }
        }

        return dataProviders;
    }

    TPrecomputedOnlineCtrData ReadPrecomputedOnlineCtrMetaData(
        const NCatboostOptions::TPoolLoadParams& loadOptions
    ) {
        CATBOOST_DEBUG_LOG << "Loading precomputed data metadata..." << Endl;

        TPrecomputedOnlineCtrData result;
        result.Meta = TPrecomputedOnlineCtrMetaData::DeserializeFromJson(
            TIFStream(loadOptions.PrecomputedMetadataFile).ReadAll()
        );
        return result;
    }

} // NCB
