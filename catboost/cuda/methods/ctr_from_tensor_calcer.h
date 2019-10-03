#pragma once

#include <catboost/cuda/cuda_lib/mapping.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/gpu_data/ctr_helper.h>

#include <catboost/private/libs/ctr_description/ctr_config.h>

namespace NCatboostCuda {
    class TCtrFromTensorCalcer {
    public:
        using TMapping = NCudaLib::TSingleMapping;
        using TVisitor = std::function<void(const TCtr&, const TCudaBuffer<float, TMapping>&, ui32)>;

        template <class TVisitor>
        TCtrFromTensorCalcer(TVisitor& ctrVisitor,
                             const THashMap<TFeatureTensor, TVector<NCB::TCtrConfig>>& ctrConfigs,
                             const TCtrTargets<TMapping>& ctrTargets)
            : Target(ctrTargets)
            , CtrConfigs(ctrConfigs)
            , CtrVisitor([&ctrVisitor](const TCtr& ctr, const TCudaBuffer<float, TMapping>& ctrValues, ui32 stream)
        {
                ctrVisitor(ctr, ctrValues, stream);
            }) {
        }

        void operator()(const TFeatureTensor& tensor,
                        TCtrBinBuilder<NCudaLib::TSingleMapping>& binBuilder);

    private:
        TVector<NCB::TCtrConfig> GetVisitOrder(const TMap<NCB::TCtrConfig, TVector<NCB::TCtrConfig>>& ctrs);

        TCalcCtrHelper<TMapping>& GetCalcCtrHelper(const TCudaBuffer<ui32, TMapping>& indices,
                                                   ui32 computationStream);

    private:
        using TCtrHelperPtr = THolder<TCalcCtrHelper<TMapping>>;
        const TCtrTargets<TMapping>& Target;
        const THashMap<TFeatureTensor, TVector<NCB::TCtrConfig>>& CtrConfigs;
        TMap<ui32, TCtrHelperPtr> CtrHelpers;
        TVisitor CtrVisitor;
    };
}
