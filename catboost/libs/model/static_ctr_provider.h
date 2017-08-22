#pragma once

#include "ctr_provider.h"
#include "model.h"

struct TStaticCtrProvider: public ICtrProvider {
public:
    TStaticCtrProvider(TCtrData&& ctrData)
        : CtrData(std::move(ctrData))
    {
        ReFill();
    }
    TStaticCtrProvider(const TCtrData& ctrData)
        : CtrData(ctrData)
    {
        ReFill();
    }

    void CalcCtrs(
        const yvector<TModelCtr>& neededCtrs,
        const NArrayRef::TConstArrayRef<ui8>& binarizedFeatures, // vector of binarized float & one hot features
        const NArrayRef::TConstArrayRef<int>& hashedCatFeatures,
        const IFeatureIndexProvider& binFeatureIndexProvider,
        size_t docCount,
        NArrayRef::TArrayRef<float> result) override;

private:
    void ReFill() {
        for (auto& ctr_value : CtrData.LearnCtrs) {
            auto& val = ctr_value.second;
            if (val.Ctr.empty()) {
                continue;
            }
            val.CtrTotal.resize(val.Ctr.size() * val.Ctr[0].size());
            auto ptr = val.CtrTotal.data();
            for (size_t i = 0; i < val.Ctr.size(); ++i) {
                for (size_t j = 0; j < val.Ctr[i].size(); ++j) {
                    *ptr = val.Ctr[i][j];
                    ++ptr;
                }
            }
            //val.Ctr.clear();
        }
    }
    TCtrData CtrData;
};
