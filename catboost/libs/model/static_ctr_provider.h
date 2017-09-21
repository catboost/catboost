#pragma once

#include "ctr_provider.h"
#include "model.h"

struct TStaticCtrProvider: public ICtrProvider {
public:
    explicit TStaticCtrProvider(TCtrData&& ctrData)
        : CtrData(std::move(ctrData))
    {}

    explicit TStaticCtrProvider(const TCtrData& ctrData)
        : CtrData(ctrData)
    {}

    void CalcCtrs(
        const yvector<TModelCtr>& neededCtrs,
        const NArrayRef::TConstArrayRef<ui8>& binarizedFeatures, // vector of binarized float & one hot features
        const NArrayRef::TConstArrayRef<int>& hashedCatFeatures,
        const IFeatureIndexProvider& binFeatureIndexProvider,
        size_t docCount,
        NArrayRef::TArrayRef<float> result) override;

private:
    TCtrData CtrData;
};
