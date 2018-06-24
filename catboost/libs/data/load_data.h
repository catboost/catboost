#pragma once

#include "pool.h"

#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/options/load_options.h>
#include <catboost/libs/options/restrictions.h>
#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data_types/pair.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/system/spinlock.h>
#include <util/stream/file.h>
#include <util/string/vector.h>

#include <util/generic/fwd.h>
#include <util/generic/set.h>

#include <string>


namespace NCB {

    class IPoolBuilder {
    public:
        virtual void Start(const TPoolMetaInfo& poolMetaInfo,
                           int docCount,
                           const TVector<int>& catFeatureIds) = 0;
        virtual void StartNextBlock(ui32 blockSize) = 0;

        virtual float GetCatFeatureValue(const TStringBuf& feature) = 0;
        virtual void AddCatFeature(ui32 localIdx, ui32 featureId, const TStringBuf& feature) = 0;
        virtual void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) = 0;
        virtual void AddAllFloatFeatures(ui32 localIdx, TConstArrayRef<float> features) = 0;
        virtual void AddTarget(ui32 localIdx, float value) = 0;
        virtual void AddWeight(ui32 localIdx, float value) = 0;
        virtual void AddQueryId(ui32 localIdx, TGroupId value) = 0;
        virtual void AddSubgroupId(ui32 localIdx, TSubgroupId value) = 0;
        virtual void AddBaseline(ui32 localIdx, ui32 offset, double value) = 0;
        virtual void AddDocId(ui32 localIdx, const TStringBuf& value) = 0;
        virtual void AddTimestamp(ui32 localIdx, ui64 value) = 0;
        virtual void SetFeatureIds(const TVector<TString>& featureIds) = 0;
        virtual void SetPairs(const TVector<TPair>& pairs) = 0;
        virtual int GetDocCount() const = 0;
        virtual TConstArrayRef<float> GetWeight() const = 0;
        virtual void GenerateDocIds(int offset) = 0;
        virtual void Finish() = 0;
        virtual ~IPoolBuilder() = default;
    };


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

}
