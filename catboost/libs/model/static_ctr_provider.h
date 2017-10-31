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

    virtual bool HasNeededCtrs(const yvector<TModelCtr>& neededCtrs) const;

    void CalcCtrs(
        const yvector<TModelCtr>& neededCtrs,
        const TConstArrayRef<ui8>& binarizedFeatures, // vector of binarized float & one hot features
        const TConstArrayRef<int>& hashedCatFeatures,
        const IFeatureIndexProvider& binFeatureIndexProvider,
        size_t docCount,
        TArrayRef<float> result) override;

private:
    TCtrData CtrData;
};
