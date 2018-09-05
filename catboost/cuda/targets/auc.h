#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>
#include <catboost/cuda/cuda_util/scan.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/helpers.h>

namespace NCatboostCuda {

    enum class  EAucType {
        Pessimistic,
        Optimistic
    };

    template <class TFloat, class TMapping>
    double ComputeAUC(const TCudaBuffer<TFloat, TMapping>& target,
                      const TCudaBuffer<TFloat, TMapping>& weights,
                      const TCudaBuffer<TFloat, TMapping>& cursor) {

        auto singleDevMapping = NCudaLib::TSingleMapping(0, target.GetObjectsSlice().Size());

        auto singleDevTarget =  TSingleBuffer<float>::Create(singleDevMapping);
        Reshard(target, singleDevTarget);

        auto singleDevCursor = TSingleBuffer<float>::Create(singleDevMapping);
        Reshard(cursor, singleDevCursor);

        auto singleDevWeights =  TSingleBuffer<float>::Create(singleDevMapping);
        Reshard(weights, singleDevWeights);


        double auc = 0;

        auto indices = TSingleBuffer<ui32>::Create(singleDevMapping);

        for (auto aucType : {EAucType::Pessimistic, EAucType::Optimistic}) {
            MakeSequence(indices);

            {
                auto tmp = TSingleBuffer<float>::CopyMapping(singleDevTarget);
                tmp.Copy(singleDevTarget);
                RadixSort(tmp, indices, aucType == EAucType::Optimistic);
            }
            {
                auto tmp = TSingleBuffer<float>::CopyMapping(singleDevCursor);
                Gather(tmp, singleDevCursor, indices);
                RadixSort(tmp, indices);
            }

            auto sortedTarget = TSingleBuffer<float>::Create(singleDevMapping);
            Gather(sortedTarget, singleDevTarget, indices);

            auto sortedWeights = TSingleBuffer<float>::Create(singleDevMapping);
            Gather(sortedWeights, singleDevWeights, indices);

            auto weightsPositive = TSingleBuffer<float>::CopyMapping(sortedWeights);
            weightsPositive.Copy(sortedWeights);
            MultiplyVector(weightsPositive, sortedTarget);

            auto prefixWeightsNegative = TSingleBuffer<float>::CopyMapping(sortedWeights);

            //swap classes
            AddVector(sortedTarget, -1.0f);
            MultiplyVector(sortedTarget, -1.0f);

            MultiplyVector(sortedWeights, sortedTarget);

            ScanVector(sortedWeights, prefixWeightsNegative, true);

            const ui32 totalObservations = prefixWeightsNegative.GetObjectsSlice().Size();
            float negativePairsWeight = 0;

            if (totalObservations) {
                TVector<float> tmp;
                prefixWeightsNegative.SliceView(TSlice(totalObservations - 1, totalObservations)).Read(tmp);
                negativePairsWeight = tmp[0];
            }

            MultiplyVector(prefixWeightsNegative, weightsPositive);

            const float correctPairsWeights = ReduceToHost(prefixWeightsNegative);
            const float positivePairsWeight = ReduceToHost(weightsPositive);

            float denum = positivePairsWeight * negativePairsWeight;
            auc += denum > 0 ? correctPairsWeights / denum : 0;
        }

        return auc / 2;
    }


}
