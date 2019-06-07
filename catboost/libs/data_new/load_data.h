#pragma once

#include "data_provider.h"
#include "loader.h"
#include "objects.h"

#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data_util/line_data_reader.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/options/load_options.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/system/types.h>


namespace NCB {
    // use from C++ code
    TDataProviderPtr ReadDataset(
        const TPathWithScheme& poolPath,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TDatasetSubset loadSubset,
        TMaybe<TVector<TString>*> classNames,
        NPar::TLocalExecutor* localExecutor
    );

    // for use from context where there's no localExecutor and proper logging handling is unimplemented
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
        TMaybe<TVector<TString>*> classNames = Nothing()
    );

    // version with explicitly specified lineReader. Only supports CatBoost dsv format
    TDataProviderPtr ReadDataset(
        THolder<ILineDataReader> lineReader,
        const TPathWithScheme& pairsFilePath, // can be uninited
        const TPathWithScheme& groupWeightsFilePath, // can be uninited
        const TPathWithScheme& baselineFilePath, // can be uninited
        const NCB::TDsvFormatOptions& poolFormat,
        const TVector<TColumn>& columnsDescription, // TODO(smirnovpavel): TVector<EColumn>
        const TVector<ui32>& ignoredFeatures,
        EObjectsOrder objectsOrder,
        TMaybe<TVector<TString>*> classNames,
        NPar::TLocalExecutor* localExecutor
    );

    TDataProviders ReadTrainDatasets(
        const NCatboostOptions::TPoolLoadParams& loadOptions,
        EObjectsOrder objectsOrder,
        bool readTestData,
        TDatasetSubset trainDatasetSubset,
        TMaybe<TVector<TString>*> classNames,
        NPar::TLocalExecutor* executor,
        TProfileInfo* profile
    );

}
