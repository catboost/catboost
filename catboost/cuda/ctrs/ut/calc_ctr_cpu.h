#pragma once

#include <util/generic/vector.h>
#include <library/containers/2d_array/2d_array.h>

namespace NCatboostCuda {
    struct TCtrStat {
        double FirstClass = 0;
        double Total = 0;
    };

    struct TCatCtrStat {
        TVector<double> FirstClass;
        double Total = 0;
    };

    struct TCpuTargetClassCtrCalcer {
        TCpuTargetClassCtrCalcer(ui32 uniqueValues,
                                 const TVector<ui32>& bins,
                                 const TVector<float>& weights,
                                 float prior,
                                 float priorDenum = 1.0)
            : UniqueValues(uniqueValues)
            , PriorNum(prior)
            , PriorDenum(priorDenum)
            , Bins(bins)
            , Weights(weights)
        {
        }

        ui32 UniqueValues;
        float PriorNum;
        float PriorDenum;
        const TVector<ui32>& Bins;
        const TVector<float>& Weights;

        template <class T>
        TArray2D<float> Calc(const TVector<ui32>& cpuIndices,
                             TVector<T>& classes,
                             ui32 numClasses) {
            TVector<TCatCtrStat> cpuStat(UniqueValues);

            TArray2D<float> ctrs(numClasses, cpuIndices.size());
            ctrs.FillZero();

            const float prior = PriorNum;
            const float denumPrior = PriorDenum;

            for (ui32 i = 0; i < cpuIndices.size(); ++i) {
                const ui32 idx = cpuIndices[i];

                const ui32 bin = Bins[idx];
                const ui32 clazz = classes[idx];

                cpuStat[bin].FirstClass.resize(numClasses);

                for (ui32 cls = 0; cls < numClasses; ++cls) {
                    auto ctr = (cpuStat[bin].FirstClass[cls] + prior) / (cpuStat[bin].Total + denumPrior);
                    ctrs[i][cls] = ctr;
                }

                cpuStat[bin].FirstClass[clazz] += Weights[idx];
                cpuStat[bin].Total += Weights[idx];
            }
            return ctrs;
        }

        inline TVector<float> ComputeFreqCtr(const TVector<ui32>* indices = nullptr) {
            TVector<TCtrStat> cpuStat(UniqueValues);
            double total = 0;
            for (ui32 i = 0; i < Bins.size(); ++i) {
                ui32 bin = Bins[i];
                cpuStat[bin].FirstClass += Weights[i];
                total += Weights[i];
            }
            TVector<float> ctr(Bins.size());
            for (ui32 i = 0; i < Bins.size(); ++i) {
                const ui32 bin = Bins[indices ? (*indices)[i] : i];
                ctr[i] = (cpuStat[bin].FirstClass + PriorNum) / (total + PriorDenum);
            }
            return ctr;
        }
    };
}
