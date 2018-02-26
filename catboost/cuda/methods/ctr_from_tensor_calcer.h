#pragma once

#include "histograms_helper.h"
#include "tree_ctrs_dataset.h"
#include <catboost/cuda/utils/countdown_latch.h>
#include <catboost/cuda/data/feature.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/models/oblivious_model.h>
#include <catboost/cuda/gpu_data/oblivious_tree_bin_builder.h>

#include <util/generic/map.h>
#include <util/generic/hash.h>
#include <util/generic/set.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/cuda_lib/device_subtasks_helper.h>

namespace NCatboostCuda {
    template <class TCtrVisitor>
    class TCtrFromTensorCalcer {
    public:
        using TMapping = NCudaLib::TSingleMapping;

        TCtrFromTensorCalcer(TCtrVisitor& ctrVisitor,
                             const THashMap<TFeatureTensor, TVector<TCtrConfig>>& ctrConfigs,
                             const TCtrTargets<TMapping>& ctrTargets)
            : Target(ctrTargets)
            , CtrConfigs(ctrConfigs)
            , CtrVisitor(ctrVisitor)
        {
        }

        void operator()(const TFeatureTensor& tensor,
                        TCtrBinBuilder<NCudaLib::TSingleMapping>& binBuilder) {
            auto profileGuard = NCudaLib::GetCudaManager().GetProfiler().Profile("calcCtrsFromTensor");
            const TCudaBuffer<ui32, TMapping>& indices = binBuilder.GetIndices();
            auto& helper = GetCalcCtrHelper(indices, binBuilder.GetStream());

            CB_ENSURE(CtrConfigs.has(tensor), "Error: unknown feature tensor");
            const auto& configs = CtrConfigs.at(tensor);

            auto grouppedConfigs = CreateEqualUpToPriorAndBinarizationCtrsGroupping(configs);

            //TODO(noxoomo): it should be done another way. we use helper implementation here
            // dirty hack for memory usage. BinFreq ctrs aren't cached in ctr-helper, so they'll stop the world after
            //calculation. But if we wouldn't drop them, then we'll use much more memory.
            //if we drop them at the end of calculation, then we can't fully overlap memcpy from host and computations
            //so visit them first, then all other
            auto ctrVisitor = [&](const TCtrConfig& ctrConfig,
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

    private:
        TVector<TCtrConfig> GetVisitOrder(const TMap<TCtrConfig, TVector<TCtrConfig>>& ctrs) {
            TVector<TCtrConfig> freqCtrs;
            TVector<TCtrConfig> restCtrs;

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

        TCalcCtrHelper<TMapping>& GetCalcCtrHelper(const TCudaBuffer<ui32, TMapping>& indices,
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

    private:
        using TCtrHelperPtr = THolder<TCalcCtrHelper<TMapping>>;
        const TCtrTargets<TMapping>& Target;
        const THashMap<TFeatureTensor, TVector<TCtrConfig>>& CtrConfigs;
        TMap<ui32, TCtrHelperPtr> CtrHelpers;
        TCtrVisitor& CtrVisitor;
    };
}
