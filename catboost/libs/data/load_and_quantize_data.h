#pragma once

#include "data_provider.h"
#include "loader.h"
#include "order.h"
#include "quantized_features_info.h"

#include <library/json/json_value.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/system/types.h>

namespace NCatboostOptions {
    struct TColumnarPoolFormatParams;
}

namespace NPar {
    class TLocalExecutor;
}

namespace NCB {
    struct TPathWithScheme;

    // use from C++ code
    TDataProviderPtr ReadAndQuantizeDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const TPathWithScheme& inputBordersPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        NJson::TJsonValue plainJsonParams,
        TMaybe<ui32> blockSize,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        TDatasetSubset loadSubset,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::TLocalExecutor* localExecutor
    );

    // for use from context where there's no localExecutor and proper logging handling is unimplemented
    TDataProviderPtr ReadAndQuantizeDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const TPathWithScheme& inputBordersPath, // can be uninited
        const NCatboostOptions::TColumnarPoolFormatParams& columnarPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        NJson::TJsonValue plainJsonParams,
        TMaybe<ui32> blockSize,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        int threadCount,
        bool verbose,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels = Nothing()
    );
}
