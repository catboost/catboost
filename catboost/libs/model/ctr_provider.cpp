#include "ctr_provider.h"
#include "static_ctr_provider.h"

#include <catboost/libs/helpers/exception.h>


TIntrusivePtr<ICtrProvider> MergeCtrProvidersData(const TVector<TIntrusivePtr<ICtrProvider>>& providers, ECtrTableMergePolicy mergePolicy) {
    TVector<const TStaticCtrProvider*> nonEmptyStaticProviders;
    for (const auto& provider : providers) {
        if (provider) {
            const TStaticCtrProvider* staticCtr = dynamic_cast<const TStaticCtrProvider*>(provider.Get());
            CB_ENSURE(staticCtr != nullptr, "only static ctr providers merging supported for now");
            nonEmptyStaticProviders.emplace_back(staticCtr);
        }
    }
    if (nonEmptyStaticProviders.empty()) {
        return TIntrusivePtr<ICtrProvider>();
    }
    if (nonEmptyStaticProviders.size() == 1) {
        return nonEmptyStaticProviders.back()->Clone();
    }
    return MergeStaticCtrProvidersData(nonEmptyStaticProviders, mergePolicy);
}
