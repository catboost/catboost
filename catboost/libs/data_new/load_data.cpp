#include "load_data.h"

#include "cb_dsv_loader.h"
#include "data_provider_builders.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/int_cast.h>
#include <catboost/libs/logging/logging.h>


namespace NCB {

    TDataProviderPtr ReadDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        NPar::TLocalExecutor* localExecutor
    ) {
        auto datasetLoader = GetProcessor<IDatasetLoader>(
            poolPath, // for choosing processor

            // processor args
            TDatasetLoaderPullArgs {
                poolPath,

                TDatasetLoaderCommonArgs {
                    pairsFilePath,
                    groupWeightsFilePath,
                    dsvPoolFormatParams.Format,
                    MakeCdProviderFromFile(dsvPoolFormatParams.CdFilePath),
                    ignoredFeatures,
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
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        int threadCount,
        bool verbose
    ) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);

        if (verbose) {
            SetVerboseLogingMode();
        } else {
            SetSilentLogingMode();
        }

        TDataProviderPtr dataProviderPtr = ReadDataset(
            poolPath,
            pairsFilePath,
            groupWeightsFilePath,
            dsvPoolFormatParams,
            ignoredFeatures,
            &localExecutor
        );

        SetVerboseLogingMode(); //TODO(smirnovpavel): verbose mode must be restored to initial

        return dataProviderPtr;
    }


    TDataProviderPtr ReadDataset(
        THolder<ILineDataReader> poolReader,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const NCB::TDsvFormatOptions& poolFormat,
        const TVector<TColumn>& columnsDescription, // TODO(smirnovpavel): TVector<EColumn>
        const TVector<ui32>& ignoredFeatures,
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
                    poolFormat,
                    MakeCdProviderFromArray(columnsDescription),
                    ignoredFeatures,
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
        bool readTestData,
        ui32 threadCount,
        TMaybe<TProfileInfo*> profile
    ) {
        loadOptions.Validate();

        /* TODO(akhropov): This cast will go away after all code is switched to new dataset interfaces.
         * MLTOOLS-140.
         */
        const TVector<ui32> ignoredFeatures = ToUnsigned(loadOptions.IgnoredFeatures);

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(threadCount - 1);

        TDataProviders dataProviders;

        if (loadOptions.LearnSetPath.Inited()) {
            dataProviders.Learn = ReadDataset(
                loadOptions.LearnSetPath,
                loadOptions.PairsFilePath,
                loadOptions.GroupWeightsFilePath,
                loadOptions.DsvPoolFormatParams,
                ignoredFeatures,
                &localExecutor
            );
            if (profile) {
                (*profile)->AddOperation("Build learn pool");
            }
        }
        dataProviders.Test.resize(0);

        if (readTestData) {
            for (int testIdx = 0; testIdx < loadOptions.TestSetPaths.ysize(); ++testIdx) {
                const NCB::TPathWithScheme& testSetPath = loadOptions.TestSetPaths[testIdx];
                const NCB::TPathWithScheme& testPairsFilePath =
                        testIdx == 0 ? loadOptions.TestPairsFilePath : NCB::TPathWithScheme();
                const NCB::TPathWithScheme& testGroupWeightsFilePath =
                        testIdx == 0 ? loadOptions.TestGroupWeightsFilePath : NCB::TPathWithScheme();

                TDataProviderPtr testDataProvider = ReadDataset(
                    testSetPath,
                    testPairsFilePath,
                    testGroupWeightsFilePath,
                    loadOptions.DsvPoolFormatParams,
                    ignoredFeatures,
                    &localExecutor
                );
                dataProviders.Test.push_back(std::move(testDataProvider));
                if (profile.Defined() && (testIdx + 1 == loadOptions.TestSetPaths.ysize())) {
                    (*profile)->AddOperation("Build test pool");
                }
            }
        }

        return dataProviders;
    }

} // NCB
