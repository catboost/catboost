#pragma once

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/data_new/loader.h>
#include <catboost/libs/data_new/data_provider_builders.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/analytical_mode_params.h>

#include <library/threading/local_executor/local_executor.h>

template <class TConsumer>
inline void ReadAndProceedPoolInBlocks(const NCB::TAnalyticalModeCommonParams& params,
                                       ui32 blockSize,
                                       TConsumer&& poolConsumer,
                                       NPar::TLocalExecutor* localExecutor) {

    auto datasetLoader = NCB::GetProcessor<NCB::IDatasetLoader>(
        params.InputPath, // for choosing processor

        // processor args
        NCB::TDatasetLoaderPullArgs {
            params.InputPath,

            NCB::TDatasetLoaderCommonArgs {
                params.PairsFilePath,
                /*GroupWeightsFilePath=*/NCB::TPathWithScheme(),
                /*BaselineFilePath=*/NCB::TPathWithScheme(),
                params.ClassNames,
                params.DsvPoolFormatParams.Format,
                MakeCdProviderFromFile(params.DsvPoolFormatParams.CdFilePath),
                /*ignoredFeatures*/ {},
                NCB::EObjectsOrder::Undefined,
                blockSize,
                NCB::TDatasetSubset::MakeColumns(),
                localExecutor
            }
        }
    );

    THolder<NCB::IDataProviderBuilder> dataProviderBuilder = NCB::CreateDataProviderBuilder(
        datasetLoader->GetVisitorType(),
        NCB::TDataProviderBuilderOptions{},
        NCB::TDatasetSubset::MakeColumns(),
        localExecutor
    );
    CB_ENSURE_INTERNAL(
        dataProviderBuilder,
        "Failed to create data provider builder for visitor of type " << datasetLoader->GetVisitorType()
    );

    NCB::IRawObjectsOrderDatasetLoader* rawObjectsOrderDatasetLoader
        = dynamic_cast<NCB::IRawObjectsOrderDatasetLoader*>(datasetLoader.Get());

    if (rawObjectsOrderDatasetLoader) {
        // process in blocks
        NCB::IRawObjectsOrderDataVisitor* visitor = dynamic_cast<NCB::IRawObjectsOrderDataVisitor*>(
            dataProviderBuilder.Get()
        );
        CB_ENSURE_INTERNAL(visitor, "failed cast of IDataProviderBuilder to IRawObjectsOrderDataVisitor");

        while (rawObjectsOrderDatasetLoader->DoBlock(visitor)) {
            poolConsumer(dataProviderBuilder->GetResult());
        }
        auto lastResult = dataProviderBuilder->GetLastResult();
        if (lastResult) {
            poolConsumer(std::move(lastResult));
        }
    } else {
        // pool is incompatible with block processing - process all pool as a whole
        datasetLoader->DoIfCompatible(dynamic_cast<NCB::IDatasetVisitor*>(dataProviderBuilder.Get()));
        poolConsumer(dataProviderBuilder->GetResult());
    }
}
