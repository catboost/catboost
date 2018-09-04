#pragma once
#include "ctr_provider.h"
#include "ctr_data.h"
#include "split.h"

#include <catboost/libs/helpers/exception.h>

#include <library/json/json_value.h>
#include <library/threading/local_executor/local_executor.h>

#include <util/system/mutex.h>


struct TStaticCtrProvider: public ICtrProvider {
public:
    TStaticCtrProvider() = default;
    explicit TStaticCtrProvider(TCtrData& ctrData)
        : CtrData(ctrData)
    {}

    bool HasNeededCtrs(const TVector<TModelCtr>& neededCtrs) const override;

    void CalcCtrs(
        const TVector<TModelCtr>& neededCtrs,
        const TConstArrayRef<ui8>& binarizedFeatures, // vector of binarized float & one hot features
        const TConstArrayRef<int>& hashedCatFeatures,
        size_t docCount,
        TArrayRef<float> result) override;

    NJson::TJsonValue ConvertCtrsToJson(const TVector<TModelCtr>& neededCtrs) const override;

    void SetupBinFeatureIndexes(
        const TVector<TFloatFeature>& floatFeatures,
        const TVector<TOneHotFeature>& oheFeatures,
        const TVector<TCatFeature>& catFeatures) override;
    bool IsSerializable() const override {
        return true;
    }
    void AddCtrCalcerData(TCtrValueTable&& valueTable) override {
        auto ctrBase = valueTable.ModelCtrBase;
        CtrData.LearnCtrs[ctrBase] = std::move(valueTable);
    }

    void Save(IOutputStream* out) const override {
        ::Save(out, CtrData);
    }

    void Load(IInputStream* inp) override {
        ::Load(inp, CtrData);
    }

    TString ModelPartIdentifier() const override {
        return "static_provider_v1";
    }

    const THashMap<TFloatSplit, TBinFeatureIndexValue>& GetFloatFeatureIndexes() const {
        return FloatFeatureIndexes;
    }

    const THashMap<int, int>& GetCatFeatureIndex() const {
        return CatFeatureIndex;
    }

    const THashMap<TOneHotSplit, TBinFeatureIndexValue>& GetOneHotFeatureIndexes() const {
        return OneHotFeatureIndexes;
    }

    ~TStaticCtrProvider() override {}
    TCtrData CtrData;
private:
    THashMap<TFloatSplit, TBinFeatureIndexValue> FloatFeatureIndexes;
    THashMap<int, int> CatFeatureIndex;
    THashMap<TOneHotSplit, TBinFeatureIndexValue> OneHotFeatureIndexes;
};

struct TStaticCtrOnFlightSerializationProvider: public ICtrProvider {
public:
    using TCtrParallelGenerator = std::function<void(const TVector<TModelCtrBase>&, TCtrDataStreamWriter*)>;

    TStaticCtrOnFlightSerializationProvider(
        TVector<TModelCtrBase> ctrBases,
        TCtrParallelGenerator ctrParallelGenerator
    )
        : CtrBases(ctrBases)
        , CtrParallelGenerator(ctrParallelGenerator)
    {
    }

    bool HasNeededCtrs(const TVector<TModelCtr>& ) const override {
        return false;
    }

    NJson::TJsonValue ConvertCtrsToJson(const TVector<TModelCtr>&) const override {
        ythrow TCatboostException() << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void CalcCtrs(
        const TVector<TModelCtr>& ,
        const TConstArrayRef<ui8>& ,
        const TConstArrayRef<int>& ,
        size_t,
        TArrayRef<float>) override {
        ythrow yexception() << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void SetupBinFeatureIndexes(
        const TVector<TFloatFeature>& ,
        const TVector<TOneHotFeature>& ,
        const TVector<TCatFeature>& ) override {
        ythrow yexception() << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }
    bool IsSerializable() const override {
        return true;
    }
    void AddCtrCalcerData(TCtrValueTable&& ) override {
        ythrow yexception() << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    void Save(IOutputStream* out) const override {
        TCtrDataStreamWriter streamWriter(out, CtrBases.size());
        CtrParallelGenerator(CtrBases, &streamWriter);
    }

    void Load(IInputStream*) override {
        ythrow yexception() << "TStaticCtrOnFlightSerializationProvider is for streamed serialization only";
    }

    TString ModelPartIdentifier() const override {
        return "static_provider_v1";
    }

    ~TStaticCtrOnFlightSerializationProvider() = default;
private:
    TVector<TModelCtrBase> CtrBases;
    TCtrParallelGenerator CtrParallelGenerator;
};

