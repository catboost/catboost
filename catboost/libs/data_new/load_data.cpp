#include "baseline.h"
#include "load_data.h"

#include "cb_dsv_loader.h"
#include "data_provider_builders.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/logging/logging.h>

#include <util/datetime/base.h>


namespace NCB {

    TDataProviderPtr ReadDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TMaybe<TVector<TString>*> classNames,
        NPar::TLocalExecutor* localExecutor
    ) {
        CB_ENSURE_INTERNAL(!baselineFilePath.Inited() || classNames, "ClassNames must be specified if baseline file is specified");
        if (classNames) {
            UpdateClassNamesFromBaselineFile(baselineFilePath, *classNames);
        }
        auto datasetLoader = GetProcessor<IDatasetLoader>(
            poolPath, // for choosing processor

            // processor args
            TDatasetLoaderPullArgs {
                poolPath,

                TDatasetLoaderCommonArgs {
                    pairsFilePath,
                    groupWeightsFilePath,
                    baselineFilePath,
                    classNames ? **classNames : TVector<TString>(),
                    dsvPoolFormatParams.Format,
                    MakeCdProviderFromFile(dsvPoolFormatParams.CdFilePath),
                    ignoredFeatures,
                    objectsOrder,
                    10000, // TODO: make it a named constant
                    localExecutor
                }
            }
        );

        THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
            datasetLoader->GetVisitorType(),
            TDataProviderBuilderOptions{},
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
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        int threadCount,
        bool verbose,
        TMaybe<TVector<TString>*> classNames
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);

        TSetLoggingVerboseOrSilent inThisScope(verbose);

        TDataProviderPtr dataProviderPtr = ReadDataset(
            poolPath,
            pairsFilePath,
            groupWeightsFilePath,
            baselineFilePath,
            dsvPoolFormatParams,
            ignoredFeatures,
            objectsOrder,
            classNames,
            &localExecutor
        );

        return dataProviderPtr;
    }


    TDataProviderPtr ReadDataset(
        THolder<ILineDataReader> poolReader,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const NCB::TDsvFormatOptions& poolFormat,
        const TVector<TColumn>& columnsDescription, // TODO(smirnovpavel): TVector<EColumn>
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TMaybe<TVector<TString>*> classNames,
        NPar::TLocalExecutor* localExecutor
    ) {
        THolder<IDataProviderBuilder> dataProviderBuilder = CreateDataProviderBuilder(
            EDatasetVisitorType::RawObjectsOrder,
            TDataProviderBuilderOptions{},
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
                    groupWeightsFilePath,
                    baselineFilePath,
                    classNames ? **classNames : TVector<TString>(),
                    poolFormat,
                    MakeCdProviderFromArray(columnsDescription),
                    ignoredFeatures,
                    objectsOrder,
                    10000, // TODO: make it a named constant
                    localExecutor
                }
            }
        );
        datasetLoader.DoIfCompatible(dynamic_cast<IDatasetVisitor*>(dataProviderBuilder.Get()));
        return dataProviderBuilder->GetResult();
    }

    TDataProviders ReadTrainDatasets(
        const NCatboostOptions::TPoolLoadParams& loadOptions,
        EObjectsOrder objectsOrder,
        bool readTestData,
        TMaybe<TVector<TString>*> classNames,
        NPar::TLocalExecutor* const executor,
        TProfileInfo* const profile
    ) {
        loadOptions.Validate();

        TDataProviders dataProviders;

        if (loadOptions.LearnSetPath.Inited()) {
            CATBOOST_DEBUG_LOG << "Loading features..." << Endl;
            auto start = Now();
            dataProviders.Learn = ReadDataset(
                loadOptions.LearnSetPath,
                loadOptions.PairsFilePath,
                loadOptions.GroupWeightsFilePath,
                loadOptions.BaselineFilePath,
                loadOptions.DsvPoolFormatParams,
                loadOptions.IgnoredFeatures,
                objectsOrder,
                classNames,
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
                const NCB::TPathWithScheme& testGroupWeightsFilePath =
                    testIdx == 0 ? loadOptions.TestGroupWeightsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testBaselineFilePath =
                    testIdx == 0 ? loadOptions.TestBaselineFilePath : NCB::TPathWithScheme();

                TDataProviderPtr testDataProvider = ReadDataset(
                    testSetPath,
                    testPairsFilePath,
                    testGroupWeightsFilePath,
                    testBaselineFilePath,
                    loadOptions.DsvPoolFormatParams,
                    loadOptions.IgnoredFeatures,
                    objectsOrder,
                    classNames,
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

} // NCB
