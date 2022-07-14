#pragma once

#include <catboost/libs/helpers/exception.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/hash.h>
#include <library/cpp/containers/2d_array/2d_array.h>

namespace NCatboostCuda {
    struct TCtrStat {
        double FirstClass = 0;
        double Total = 0;
    };

    struct TCatCtrStat {
        TVector<double> FirstClass;
        double Total = 0;

        TCatCtrStat& operator+=(const TCatCtrStat& other) {
            if (FirstClass.size() < other.FirstClass.size()) {
                FirstClass.resize(other.FirstClass.size());
            }
            if (!other.FirstClass.empty()) {
                CB_ENSURE(other.FirstClass.size() == FirstClass.size());
                for (ui64 i = 0; i < FirstClass.size(); ++i) {
                    FirstClass[i] += other.FirstClass[i];
                }
                Total += other.Total;
            }
            return *this;
        }
    };

    struct TCpuTargetClassCtrCalcer {
        TCpuTargetClassCtrCalcer(ui32 uniqueValues,
                                 const TVector<ui32>& bins,
                                 TConstArrayRef<float> weights,
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
        TConstArrayRef<float> Weights;

        template <class T, class TGid>
        TArray2D<float> Calc(const TVector<ui32>& cpuIndices,
                             TConstArrayRef<TGid> groupIds,
                             TVector<T>& classes,
                             ui32 numClasses) {
            TVector<TCatCtrStat> cpuStat(UniqueValues);
            THashMap<ui32, TCatCtrStat> currentGroup;

            TArray2D<float> ctrs(numClasses, cpuIndices.size());
            ctrs.FillZero();

            const float prior = PriorNum;
            const float denumPrior = PriorDenum;

            for (ui32 i = 0; i < cpuIndices.size(); ++i) {
                const ui32 idx = cpuIndices[i];

                const ui32 bin = Bins[idx];
                const ui32 clazz = classes[idx];

                cpuStat[bin].FirstClass.resize(numClasses);
                currentGroup[bin].FirstClass.resize(numClasses);

                for (ui32 cls = 0; cls < numClasses; ++cls) {
                    auto ctr = (cpuStat[bin].FirstClass[cls] + prior) / (cpuStat[bin].Total + denumPrior);
                    ctrs[i][cls] = ctr;
                }

                currentGroup[bin].FirstClass[clazz] += Weights[idx];
                currentGroup[bin].Total += Weights[idx];

                auto groupId = groupIds[cpuIndices[i]];
                auto nextGroupId = (i + 1) < cpuIndices.size() ? groupIds[cpuIndices[i + 1]] : groupIds.size() + 10000;
                if (groupId != nextGroupId) {
                    for (const auto& [b, ctr] : currentGroup) {
                        cpuStat[b] += ctr;
                    }
                    currentGroup.clear();
                }
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
