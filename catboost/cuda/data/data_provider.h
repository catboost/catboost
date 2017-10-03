#pragma once

#include "columns.h"
#include "binarization_config.h"
#include <util/generic/yexception.h>

class TDataProvider: public TMoveOnly {
public:
    TDataProvider() {
    }

    bool IsEmpty() const {
        return GetSampleCount() == 0;
    }

    size_t GetEffectiveFeatureCount() const {
        return Features.size();
    }

    bool HasFeatureId(ui32 featureId) const {
        return IndicesToLocalIndicesRemap.has(featureId);
    }

    const IFeatureValuesHolder& GetFeatureById(ui32 featureId) const {
        const ui32 localId = IndicesToLocalIndicesRemap.at(featureId);
        CB_ENSURE(Features[localId], "Error: nullptr feature found. something wrong");
        return *Features[localId];
    }

    const IFeatureValuesHolder& GetFeatureByIndex(ui32 index) const {
        CB_ENSURE(Features[index], "Error: nullptr feature found. something wrong");
        return *Features[index];
    }

    const TBinarizedFloatValuesHolder& GetBinarizedFloatFeatureById(ui32 id) const {
        auto index = IndicesToLocalIndicesRemap.at(id);
        CB_ENSURE(Features[index], "Error: nullptr feature found. something wrong");
        return dynamic_cast<const TBinarizedFloatValuesHolder&>(*Features[index]);
    }

    size_t GetSampleCount() const {
        return Targets.size();
    }

    TDataProvider const* Get() const {
        if (IsEmpty()) {
            return nullptr;
        }
        return this;
    }

    const yvector<float>& GetTargets() const {
        return Targets;
    }

    const yvector<float>& GetWeights() const {
        return Weights;
    }

    const yvector<int>& GetQueryIds() const {
        if (QueryIds.size() != Targets.size()) {
            ythrow yexception() << "Don't have query ids: qids vector size is less, than points (target) size. If you need qids, load data with LF_QUERY_ID flag";
        }
        return QueryIds;
    }

    const yvector<yvector<ui32>>& GetQueries() const {
        if (QueryIds.size() != Targets.size()) {
            ythrow yexception() << "Don't store queries: qids vector size is less, than points (target) size. If you need qids, load data with LF_QUERY_ID flag";
        }
        return Queries;
    }

    ui32 GetQidByLine(size_t line) const {
        return GetQueryIds()[GetQueries()[line][0]];
    }

    const yvector<TString>& GetFeatureNames() const {
        return FeatureNames;
    }

    const yset<int>& GetCatFeatureIds() const {
        return CatFeatureIds;
    }

    bool HasBaseline() const {
        return Baseline.size() && Baseline[0].size() == GetSampleCount();
    }

    const yvector<float>& GetBaseline() const {
        return Baseline[0];
    }

private:
    yvector<TFeatureColumnPtr> Features;

    yvector<ui32> Order;
    yvector<int> QueryIds;
    yvector<yvector<ui32>> Queries;

    yvector<ui32> DocIds;
    yvector<float> Targets;
    yvector<float> Weights;
    yvector<yvector<float>> Baseline;

    ymap<ui32, ui32> IndicesToLocalIndicesRemap;

    friend class TCatBoostProtoPoolReader;

    void BuildIndicesRemap() {
        IndicesToLocalIndicesRemap.clear();

        for (ui32 i = 0; i < Features.size(); ++i) {
            CB_ENSURE(Features[i], "Error: nullptr feature found. something wrong");
            IndicesToLocalIndicesRemap[Features[i]->GetId()] = i;
        }
    }

    //for cpu model conversion
    yvector<TString> FeatureNames;
    yset<int> CatFeatureIds;

    friend class TDataProviderBuilder;
};
