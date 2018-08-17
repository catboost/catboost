#pragma once

#include "pool.h"

#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/pool_builder/pool_builder.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/maybe.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

namespace NCB {

    THolder<IPoolBuilder> InitBuilder(
        const NCB::TPathWithScheme& poolPath, // quantize, if scheme == "quantized"
        const NPar::TLocalExecutor& localExecutor,
        TPool* pool);

    // add target converter to ReadPool for processing target labels
    class TTargetConverter {
    public:
        TTargetConverter(const EConvertTargetPolicy readingPoolTargetPolicy,
                         const TVector<TString>& inputClassNames,
                         TVector<TString>* const outputClassNames);

        float ConvertLabel(const TStringBuf& label) const;
        float ProcessLabel(const TString& label);

        TVector<float> PostprocessLabels(TConstArrayRef<TString> labels);
        void SetOutputClassNames() const;

        EConvertTargetPolicy GetTargetPolicy() const;

        const TVector<TString>& GetInputClassNames() const;
    private:
        const EConvertTargetPolicy TargetPolicy;
        const TVector<TString>& InputClassNames;
        TVector<TString>* const OutputClassNames;
        THashMap<TString, int> LabelToClass; // used with targetPolicy = MakeClassNames/UseClassNames
    };

    TTargetConverter MakeTargetConverter(const TVector<TString>& classNames);

    void ReadPool(const TPathWithScheme& poolPath,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const TPathWithScheme& groupWeightsFilePath, // can be uninited
                  const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
                  const TVector<int>& ignoredFeatures,
                  int threadCount,
                  bool verbose,
                  TPool* pool);

    void ReadPool(THolder<ILineDataReader> poolReader,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const TPathWithScheme& groupWeightsFilePath, // can be uninited
                  const NCB::TDsvFormatOptions& poolFormat,
                  const TVector<TColumn>& columnsDescription, // TODO(smirnovpavel): TVector<EColumn>
                  const TVector<int>& ignoredFeatures,
                  const TVector<TString>& classNames,
                  NPar::TLocalExecutor* localExecutor,
                  TPool* pool);

    void ReadPool(const TPathWithScheme& poolPath,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const TPathWithScheme& groupWeightsFilePath, // can be uninited
                  const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
                  const TVector<int>& ignoredFeatures,
                  int threadCount,
                  bool verbose,
                  TTargetConverter* const targetConverter,
                  TPool* pool);

    void ReadPool(const TPathWithScheme& poolPath,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const TPathWithScheme& groupWeightsFilePath, // can be uninited
                  const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
                  const TVector<int>& ignoredFeatures,
                  bool verbose,
                  TTargetConverter* const targetConverter,
                  NPar::TLocalExecutor* localExecutor,
                  IPoolBuilder* poolBuilder);

    void ReadPool(const TPathWithScheme& poolPath,
                  const TPathWithScheme& pairsFilePath, // can be uninited
                  const TPathWithScheme& groupWeightsFilePath, // can be uninited
                  const NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
                  int threadCount,
                  bool verbose,
                  IPoolBuilder& poolBuilder);

    void ReadTrainPools(const NCatboostOptions::TPoolLoadParams& loadOptions,
                        bool readTestData,
                        int threadCount,
                        TTargetConverter* const targetConverter,
                        TMaybe<TProfileInfo*> profile,
                        TTrainPools* trainPools);
}
