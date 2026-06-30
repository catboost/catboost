#pragma once

#include "ctrs.h"
#include "data_provider.h"
#include "loader.h"
#include "objects.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/private/libs/data_util/line_data_reader.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/options/load_options.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NJson {
    class TJsonValue;
}


namespace NCB {
    // use from C++ code
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
    );

    // for use from context where there's no localExecutor and proper logging handling is unimplemented
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
        TMaybe<TVector<NJson::TJsonValue>*> classLabels = Nothing()
    );

    // version with explicitly specified lineReader. Only supports CatBoost dsv format
    TDataProviderPtr ReadDataset(
        THolder<ILineDataReader>&& lineReader,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& graphFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& timestampsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const TPathWithScheme& featureNamesPath, // can be uninited
        const TPathWithScheme& poolMetaInfoPath, // can be uninited
        const NCB::TDsvFormatOptions& poolFormat,
        const TVector<TColumn>& columnsDescription, // TODO(smirnovpavel): TVector<EColumn>
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        bool loadSampleIds,
        bool forceUnitAutoPairWeights,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::ILocalExecutor* localExecutor
    );

    TDataProviders ReadTrainDatasets(
        TMaybe<ETaskType> taskType,
        const NCatboostOptions::TPoolLoadParams& loadOptions,
        EObjectsOrder objectsOrder,
        bool readTestData,
        TDatasetSubset learnDatasetSubset,
        TConstArrayRef<TDatasetSubset> testDatasetSubsets,
        bool forceUnitAutoPairWeights,
        TMaybe<TVector<NJson::TJsonValue>*> classLabels,
        NPar::ILocalExecutor* executor,
        TProfileInfo* profile
    );

    TPrecomputedOnlineCtrData ReadPrecomputedOnlineCtrMetaData(
        const NCatboostOptions::TPoolLoadParams& loadOptions
    );

}
