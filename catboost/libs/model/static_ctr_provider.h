#pragma once

#include <util/system/mutex.h>
#include <library/threading/local_executor/local_executor.h>
#include <catboost/libs/helpers/exception.h>
#include "ctr_provider.h"
#include "ctr_data.h"
#include "split.h"

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

    void SetupBinFeatureIndexes(
        const TVector<TFloatFeature>& floatFeatures,
        const TVector<TOneHotFeature>& oheFeatures,
        const TVector<TCatFeature>& catFeatures) override {
        int currentIndex = 0;
        FloatFeatureIndexes.clear();
        for (const auto& floatFeature : floatFeatures) {
            for (const auto& border : floatFeature.Borders) {
                TFloatSplit split{floatFeature.FeatureIndex, border};
                FloatFeatureIndexes[split] = currentIndex;
                ++currentIndex;
            }
        }
        OneHotFeatureIndexes.clear();
        for (const auto& oheFeature : oheFeatures) {
            for (int valueId = 0; valueId < oheFeature.Values.ysize(); ++valueId) {
                TOneHotSplit feature{oheFeature.CatFeatureIndex, oheFeature.Values[valueId]};
                OneHotFeatureIndexes[feature] = currentIndex;
                ++currentIndex;
            }
        }
        CatFeatureIndex.clear();
        for (const auto& catFeature : catFeatures) {
            const int prevSize = CatFeatureIndex.ysize();
            CatFeatureIndex[catFeature.FeatureIndex] = prevSize;
        }
    }
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

    ~TStaticCtrProvider() override {}
    TCtrData CtrData;
private:
    yhash<TFloatSplit, int> FloatFeatureIndexes;
    yhash<int, int> CatFeatureIndex;
    yhash<TOneHotSplit, int> OneHotFeatureIndexes;
};

struct TStaticCtrOnFlightSerializationProvider: public ICtrProvider {
public:
    TStaticCtrOnFlightSerializationProvider(
        TVector<TModelCtrBase> ctrBases,
        std::function<TCtrValueTable(const TModelCtrBase&)> ctrTableGenerator,
        NPar::TLocalExecutor& localExecutor)
        : CtrBases(ctrBases)
        , CtrTableGenerator(ctrTableGenerator)
        , LocalExecutor(localExecutor)

    {
    }

    bool HasNeededCtrs(const TVector<TModelCtr>& ) const override {
        return false;
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
        LocalExecutor.ExecRange([this, &streamWriter] (int i) {
            streamWriter.SaveOneCtr(
                CtrTableGenerator(CtrBases[i])
            );
        }, 0, CtrBases.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
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
    std::function<TCtrValueTable(const TModelCtrBase&)> CtrTableGenerator;
    NPar::TLocalExecutor& LocalExecutor;
};

