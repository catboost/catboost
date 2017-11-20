#pragma once

#include "columns.h"
#include "binarization_config.h"
#include <util/generic/yexception.h>

namespace NCatboostCuda
{
    class TDataProvider: public TMoveOnly
    {
    public:
        TDataProvider() = default;

        bool IsEmpty() const
        {
            return GetSampleCount() == 0;
        }

        void SetShuffleSeed(ui64 seed)  {
            IsShuffledFlag = true;
            ShuffleSeed = seed;

        }
        ui64 GetShuffleSeed() const {
            return ShuffleSeed;
        }

        size_t GetEffectiveFeatureCount() const
        {
            return Features.size();
        }

        bool HasFeatureId(ui32 featureId) const
        {
            return IndicesToLocalIndicesRemap.has(featureId);
        }

        const IFeatureValuesHolder& GetFeatureById(ui32 featureId) const
        {
            if (!IndicesToLocalIndicesRemap.has(featureId))
            {
                ythrow TCatboostException() << "No feature with feature id #" << featureId << " found";
            }
            const ui32 localId = IndicesToLocalIndicesRemap.at(featureId);
            CB_ENSURE(Features[localId], "Error: nullptr feature found. something wrong");
            return *Features[localId];
        }

        const IFeatureValuesHolder& GetFeatureByIndex(ui32 index) const
        {
            CB_ENSURE(Features[index], "Error: nullptr feature found. something wrong");
            return *Features[index];
        }

        const TBinarizedFloatValuesHolder& GetBinarizedFloatFeatureById(ui32 id) const
        {
            auto index = IndicesToLocalIndicesRemap.at(id);
            CB_ENSURE(Features[index], "Error: nullptr feature found. something wrong");
            return dynamic_cast<const TBinarizedFloatValuesHolder&>(*Features[index]);
        }

        size_t GetSampleCount() const
        {
            return Targets.size();
        }

        TDataProvider const* Get() const
        {
            if (IsEmpty())
            {
                return nullptr;
            }
            return this;
        }

        const TVector<float>& GetTargets() const
        {
            return Targets;
        }

        const TVector<float>& GetWeights() const
        {
            return Weights;
        }

        const TVector<ui32>& GetQueryIds() const
        {
            if (QueryIds.size() != Targets.size())
            {
                ythrow yexception()
                        << "Don't have query ids: qids vector size is less, than points (target) size. If you need qids, load data with LF_QUERY_ID flag";
            }
            return QueryIds;
        }

        const TVector<TVector<ui32>>& GetQueries() const
        {
            if (QueryIds.size() != Targets.size())
            {
                ythrow yexception()
                        << "Don't store queries: qids vector size is less, than points (target) size. If you need qids, load data with LF_QUERY_ID flag";
            }
            return Queries;
        }

        ui32 GetQidByLine(size_t line) const
        {
            return GetQueryIds()[GetQueries()[line][0]];
        }

        const TVector<TString>& GetFeatureNames() const
        {
            return FeatureNames;
        }

        const yset<int>& GetCatFeatureIds() const
        {
            return CatFeatureIds;
        }

        bool HasBaseline() const
        {
            return Baseline.size() && Baseline[0].size() == GetSampleCount();
        }

        const TVector<float>& GetBaseline() const
        {
            return Baseline[0];
        }

    private:
        TVector<TFeatureColumnPtr> Features;

        TVector<ui32> Order;
        TVector<ui32> QueryIds;
        TVector<TVector<ui32>> Queries;

        TVector<ui32> DocIds;
        TVector<float> Targets;
        TVector<float> Weights;
        TVector<TVector<float>> Baseline;

        ymap<ui32, ui32> IndicesToLocalIndicesRemap;

        void BuildIndicesRemap()
        {
            IndicesToLocalIndicesRemap.clear();

            for (ui32 i = 0; i < Features.size(); ++i)
            {
                CB_ENSURE(Features[i], "Error: nullptr feature found. something wrong");
                IndicesToLocalIndicesRemap[Features[i]->GetId()] = i;
            }
        }

        //for cpu model conversion
        TVector<TString> FeatureNames;
        yset<int> CatFeatureIds;

        ui64 ShuffleSeed = 0;
        bool IsShuffledFlag = false;

        friend class TDataProviderBuilder;

        friend class TCpuPoolBasedDataProviderBuilder;

        friend class TCatBoostProtoPoolReader;
    };
}
