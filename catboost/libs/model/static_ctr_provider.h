#pragma once
#include "ctr_provider.h"
#include "ctr_data.h"
#include "split.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/hash.h>
#include <util/generic/utility.h>

#include <functional>


class TStaticCtrProvider: public ICtrProvider {
public:
    TStaticCtrProvider() = default;
    explicit TStaticCtrProvider(TCtrData& ctrData)
        : CtrData(ctrData)
    {}
    ~TStaticCtrProvider() override {}

    bool HasNeededCtrs(TConstArrayRef<TModelCtr> neededCtrs) const override;

    void CalcCtrs(
        const TConstArrayRef<TModelCtr> neededCtrs,
        const TConstArrayRef<ui8> binarizedFeatures, // vector of binarized float & one hot features
        const TConstArrayRef<ui32> hashedCatFeatures,
        size_t docCount,
        TArrayRef<float> result) override;

    void SetupBinFeatureIndexes(
        const TConstArrayRef<TFloatFeature> floatFeatures,
        const TConstArrayRef<TOneHotFeature> oheFeatures,
        const TConstArrayRef<TCatFeature> catFeatures) override;
    bool IsSerializable() const override {
        return true;
    }

    void AddCtrCalcerData(TCtrValueTable&& valueTable) override {
        auto ctrBase = valueTable.ModelCtrBase;
        CtrData.LearnCtrs[ctrBase] = std::move(valueTable);
    }

    void DropUnusedTables(TConstArrayRef<TModelCtrBase> usedModelCtrBase) override {
        TCtrData ctrData;
        for (auto& base: usedModelCtrBase) {
            ctrData.LearnCtrs[base] = std::move(CtrData.LearnCtrs[base]);
        }
        DoSwap(CtrData, ctrData);
    }

    void Save(IOutputStream* out) const override {
        ::Save(out, CtrData);
    }

    void Load(IInputStream* inp) override {
        ::Load(inp, CtrData);
    }

    void LoadNonOwning(TMemoryInput* in) {
        CtrData.LoadNonOwning(in);
    }

    static TString ModelPartId() {
        return "static_provider_v1";
    }

    TString ModelPartIdentifier() const override {
        return ModelPartId();
    }

    const THashMap<TFloatSplit, TBinFeatureIndexValue>& GetFloatFeatureIndexes() const {
        return FloatFeatureIndexes;
    }

    const THashMap<TOneHotSplit, TBinFeatureIndexValue>& GetOneHotFeatureIndexes() const {
        return OneHotFeatureIndexes;
    }

    virtual TIntrusivePtr<ICtrProvider> Clone() const override;

public:
    TCtrData CtrData;
private:
    THashMap<TFloatSplit, TBinFeatureIndexValue> FloatFeatureIndexes;
    THashMap<int, int> CatFeatureIndex;
    THashMap<TOneHotSplit, TBinFeatureIndexValue> OneHotFeatureIndexes;
};

class TStaticCtrOnFlightSerializationProvider: public ICtrProvider {
public:
    using TCtrParallelGenerator = std::function<void(const TVector<TModelCtrBase>&, TCtrDataStreamWriter*)>;

public:
    TStaticCtrOnFlightSerializationProvider(
        TVector<TModelCtrBase> ctrBases,
        TCtrParallelGenerator ctrParallelGenerator
    )
        : CtrBases(ctrBases)
        , CtrParallelGenerator(ctrParallelGenerator)
    {
    }
    ~TStaticCtrOnFlightSerializationProvider() = default;

    bool HasNeededCtrs(const TConstArrayRef<TModelCtr>) const override {
        return false;
    }

    void CalcCtrs(
        const TConstArrayRef<TModelCtr>,
        const TConstArrayRef<ui8>,
        const TConstArrayRef<ui32>,
        size_t,
        TArrayRef<float>) override {

        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void SetupBinFeatureIndexes(
        const TConstArrayRef<TFloatFeature>,
        const TConstArrayRef<TOneHotFeature>,
        const TConstArrayRef<TCatFeature>) override {

        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }
    bool IsSerializable() const override {
        return true;
    }
    void AddCtrCalcerData(TCtrValueTable&& ) override {
        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void DropUnusedTables(TConstArrayRef<TModelCtrBase>) override {
        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void Save(IOutputStream* out) const override {
        TCtrDataStreamWriter streamWriter(out, CtrBases.size());
        CtrParallelGenerator(CtrBases, &streamWriter);
    }

    void Load(IInputStream*) override {
        ythrow TCatBoostException()
            << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    TString ModelPartIdentifier() const override {
        return "static_provider_v1";
    }

private:
    TVector<TModelCtrBase> CtrBases;
    TCtrParallelGenerator CtrParallelGenerator;
};

TIntrusivePtr<TStaticCtrProvider> MergeStaticCtrProvidersData(
    const TVector<const TStaticCtrProvider*>& providers,
    ECtrTableMergePolicy mergePolicy);
