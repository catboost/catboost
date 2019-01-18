#include "ctr_from_tensor_calcer.h"
#include <catboost/cuda/gpu_data/kernels.h>

namespace NCatboostCuda {
    void TCtrFromTensorCalcer::operator()(const TFeatureTensor& tensor, TCtrBinBuilder<NCudaLib::TSingleMapping>& binBuilder) {
        auto profileGuard = NCudaLib::GetCudaManager().GetProfiler().Profile("calcCtrsFromTensor");
        const TCudaBuffer<ui32, TMapping>& indices = binBuilder.GetIndices();
        auto& helper = GetCalcCtrHelper(indices, binBuilder.GetStream());

        CB_ENSURE(CtrConfigs.contains(tensor), "Error: unknown feature tensor");
        const auto& configs = CtrConfigs.at(tensor);

        auto grouppedConfigs = CreateEqualUpToPriorAndBinarizationCtrsGroupping(configs);

        //TODO(noxoomo): it should be done another way. we use helper implementation here
        // dirty hack for memory usage. BinFreq ctrs aren't cached in ctr-helper, so they'll stop the world after
        //calculation. But if we wouldn't drop them, then we'll use much more memory.
        //if we drop them at the end of calculation, then we can't fully overlap memcpy from host and computations
        //so visit them first, then all other
        TCtrVisitor<NCudaLib::TSingleMapping> ctrVisitor = [&](const NCB::TCtrConfig& ctrConfig,
                                                               const TCudaBuffer<float, TMapping>& ctrValues,
                                                               ui32 stream) {
            TCtr ctr(tensor, ctrConfig);
            CtrVisitor(ctr, ctrValues, stream);
        };

        auto visitOrder = GetVisitOrder(grouppedConfigs);

        for (const auto& configWithoutPrior : visitOrder) {
            if (configWithoutPrior.Type == ECtrType::FeatureFreq) { //faster impl for special weights type (all 1.0). TODO(noxoomo): check correctness
                binBuilder.VisitEqualUpToPriorFreqCtrs(grouppedConfigs[configWithoutPrior], ctrVisitor);
            } else {
                helper.VisitEqualUpToPriorCtrs(grouppedConfigs[configWithoutPrior], ctrVisitor);
            }
        }
    }

    TVector<NCB::TCtrConfig> TCtrFromTensorCalcer::GetVisitOrder(const TMap<NCB::TCtrConfig, TVector<NCB::TCtrConfig>>& ctrs) {
        TVector<NCB::TCtrConfig> freqCtrs;
        TVector<NCB::TCtrConfig> restCtrs;

        for (auto& entry : ctrs) {
            if (entry.first.Type == ECtrType::FeatureFreq) {
                freqCtrs.push_back(entry.first);
            } else {
                restCtrs.push_back(entry.first);
            }
        }

        for (auto rest : restCtrs) {
            freqCtrs.push_back(rest);
        }
        return freqCtrs;
    }

    TCalcCtrHelper<TCtrFromTensorCalcer::TMapping>& TCtrFromTensorCalcer::GetCalcCtrHelper(const TCudaBuffer<ui32, TCtrFromTensorCalcer::TMapping>& indices,
                                                                                           ui32 computationStream) {
        if (CtrHelpers.count(computationStream) == 0) {
            CtrHelpers[computationStream] = MakeHolder<TCalcCtrHelper<TMapping>>(Target,
                                                                                 indices,
                                                                                 computationStream);
        } else {
            CtrHelpers[computationStream]->Reset(indices);
        }
        return *CtrHelpers[computationStream];
    }

}
