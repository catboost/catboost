#pragma once

#include "data_provider_builders.h"
#include "loader.h"

#include <catboost/libs/column_description/cd_parser.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/dataset_reading_params.h>


#include <library/threading/local_executor/local_executor.h>

template <class TConsumer>
inline void ReadAndProceedPoolInBlocks(const NCatboostOptions::TDatasetReadingParams& params,
                                       ui32 blockSize,
                                       TConsumer&& poolConsumer,
                                       NPar::TLocalExecutor* localExecutor) {

    auto datasetLoader = NCB::GetProcessor<NCB::IDatasetLoader>(
        params.PoolPath, // for choosing processor

        // processor args
        NCB::TDatasetLoaderPullArgs {
            params.PoolPath,

            NCB::TDatasetLoaderCommonArgs {
                params.PairsFilePath,
                /*GroupWeightsFilePath=*/NCB::TPathWithScheme(),
                /*BaselineFilePath=*/NCB::TPathWithScheme(),
                /*TimestampsFilePath*/NCB::TPathWithScheme(),
                params.FeatureNamesPath,
                params.ClassLabels,
                params.ColumnarPoolFormatParams.DsvFormat,
                MakeCdProviderFromFile(params.ColumnarPoolFormatParams.CdFilePath),
                params.IgnoredFeatures,
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
            auto result = dataProviderBuilder->GetResult();
            if (result) {
                poolConsumer(std::move(result));
            }
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
