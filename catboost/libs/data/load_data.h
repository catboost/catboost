#pragma once

#include "pool.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/pool_builder/pool_builder.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

namespace NCB {

    THolder<IPoolBuilder> InitBuilder(const NPar::TLocalExecutor& localExecutor, TPool* pool);

    void ReadPool(const TPathWithScheme& poolPath,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
                  const TVector<int>& ignoredFeatures,
                  int threadCount,
                  bool verbose,
                  const TVector<TString>& classNames,
                  TPool* pool);

    void ReadPool(const TPathWithScheme& poolPath,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
                  const TVector<int>& ignoredFeatures,
                  bool verbose,
                  const TVector<TString>& classNames,
                  NPar::TLocalExecutor* localExecutor,
                  IPoolBuilder* poolBuilder);

    void ReadPool(const TPathWithScheme& poolPath,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
                  int threadCount,
                  bool verbose,
                  IPoolBuilder& poolBuilder);

    void ReadTrainPools(const NCatboostOptions::TPoolLoadParams& loadOptions,
                        bool readTestData,
                        int threadCount,
                        const TVector<TString>& classNames,
                        TMaybe<TProfileInfo*> profile,
                        TTrainPools* trainPools);
}
