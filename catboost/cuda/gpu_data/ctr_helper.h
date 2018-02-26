#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/ctrs/ctr_bins_builder.h>
#include <catboost/cuda/ctrs/ctr_calcers.h>

namespace NCatboostCuda {
    template <class TMapping>
    struct TCtrTargets {
        TCudaBuffer<float, TMapping> WeightedTarget;
        TCudaBuffer<ui8, TMapping> BinarizedTarget;
        TCudaBuffer<float, TMapping> Weights;

        //zero for test, 1 for learn
        bool IsTrivialWeights() const {
            return true;
        }

        float TotalWeight;

        TSlice LearnSlice;
        TSlice TestSlice;
    };

    inline TCtrTargets<NCudaLib::TSingleMapping> DeviceView(const TCtrTargets<NCudaLib::TMirrorMapping>& mirrorTargets, ui32 devId) {
        TCtrTargets<NCudaLib::TSingleMapping> view;
        view.WeightedTarget = mirrorTargets.WeightedTarget.DeviceView(devId);
        view.BinarizedTarget = mirrorTargets.BinarizedTarget.DeviceView(devId);
        view.Weights = mirrorTargets.Weights.DeviceView(devId);

        view.TotalWeight = mirrorTargets.TotalWeight;
        view.LearnSlice = mirrorTargets.LearnSlice;
        view.TestSlice = mirrorTargets.TestSlice;
        return view;
    }

    template <class TMapping>
    class TCalcCtrHelper: public TNonCopyable {
    public:
        template <class TUi32>
        TCalcCtrHelper(const TCtrTargets<TMapping>& target,
                       const TCudaBuffer<TUi32, TMapping>& sortedByBinIndices,
                       ui32 stream = 0)
            : CtrTargets(target)
            , SortedByBinsIndices(sortedByBinIndices.ConstCopyView())
            , Stream(stream)
        {
        }

        void UseFullDataForCatFeatureStats(bool flag) {
            TCalcCtrHelper::ComputeCatFeatureStatOnFullData = flag;
        }

        template <class TUi32>
        void Reset(const TCudaBuffer<TUi32, TMapping>& sortedByBinIndices) {
            SortedByBinsIndices = sortedByBinIndices.ConstCopyView();
            if (HistoryCalcer) {
                if (CtrTargets.IsTrivialWeights()) {
                    HistoryCalcer->Reset(sortedByBinIndices, static_cast<ui32>(CtrTargets.LearnSlice.Size()));
                } else {
                    HistoryCalcer->Reset(GetWeights(), sortedByBinIndices);
                }
            }
        }

        template <class TVisitor>
        inline TCalcCtrHelper& VisitEqualUpToPriorCtrs(const TVector<TCtrConfig>& configs,
                                                       TVisitor&& visitor) {
            for (auto& config : configs) {
                CB_ENSURE(IsEqualUpToPriorAndBinarization(config, configs[0]), "Error: could visit only one-type ctrs only");
            }
            auto ctrType = configs[0].Type;
            const ui32 mask = TCtrBinBuilder<TMapping>::GetMask();
            auto weights = GetWeights();
            auto createHistoryCalcer = [&]() {
                if (CtrTargets.IsTrivialWeights()) {
                    HistoryCalcer.Reset(new THistoryBasedCtrCalcer<TMapping>(SortedByBinsIndices,
                                                                             CtrTargets.LearnSlice.Size(),
                                                                             mask,
                                                                             Stream));
                } else {
                    HistoryCalcer.Reset(new THistoryBasedCtrCalcer<TMapping>(weights,
                                                                             SortedByBinsIndices,
                                                                             mask,
                                                                             Stream));
                }
            };

            if (IsCatFeatureStatisticCtr(ctrType)) {
                if (ComputeCatFeatureStatOnFullData) {
                    TCtrBinBuilder<TMapping> binBuilder(Stream);
                    binBuilder.SetIndices(SortedByBinsIndices, CtrTargets.LearnSlice);
                    //

                    binBuilder.VisitEqualUpToPriorFreqCtrs(configs,
                                                           std::forward<TVisitor>(visitor));
                } else {
                    TWeightedBinFreqCalcer<TMapping> weightedFreqCalcer(weights,
                                                                        CtrTargets.TotalWeight,
                                                                        mask,
                                                                        Stream);
                    weightedFreqCalcer.VisitEqualUpToPriorFreqCtrs(SortedByBinsIndices,
                                                                   configs,
                                                                   std::forward<TVisitor>(visitor));
                }
            } else if (IsBinarizedTargetCtr(ctrType)) {
                if (!HistoryCalcer) {
                    createHistoryCalcer();
                }
                if (!HistoryCalcer->HasBinarizedTargetSample()) {
                    HistoryCalcer->SetBinarizedSample(CtrTargets.BinarizedTarget.SliceView(weights.GetObjectsSlice()));
                }
                HistoryCalcer->VisitCatFeatureCtr(configs, std::forward<TVisitor>(visitor));
            } else {
                CB_ENSURE(IsFloatTargetCtr(configs[0].Type));
                if (!HistoryCalcer) {
                    createHistoryCalcer();
                }
                if (!HistoryCalcer->HasFloatTargetSample()) {
                    HistoryCalcer->SetFloatSample(GetWeightedTarget());
                }
                HistoryCalcer->VisitFloatFeatureMeanCtrs(configs, std::forward<TVisitor>(visitor));
            }
            return *this;
        }

        inline TCalcCtrHelper& ComputeCtr(const TCtrConfig& config,
                                          TCudaBuffer<float, TMapping>& dst) {
            return VisitEqualUpToPriorCtrs(SingletonVector(config),
                                           [&](const TCtrConfig& ctrConfig,
                                               const TCudaBuffer<float, TMapping>& ctr,
                                               ui32 stream) {
                                               CB_ENSURE(ctrConfig == config);
                                               dst.Reset(ctr.GetMapping());
                                               dst.Copy(ctr, stream);
                                           });
        };

        inline TCudaBuffer<float, TMapping> ComputeCtr(const TCtrConfig& config) {
            TCudaBuffer<float, TMapping> floatCtr;
            ComputeCtr(config, floatCtr);
            return floatCtr;
        };

    private:
        TCudaBuffer<const float, TMapping> GetWeights() {
            return CtrTargets.Weights.SliceView(SortedByBinsIndices.GetObjectsSlice());
        }

        TCudaBuffer<const float, TMapping> GetWeightedTarget() {
            return CtrTargets.WeightedTarget.SliceView(SortedByBinsIndices.GetObjectsSlice());
        }

    private:
        const TCtrTargets<TMapping>& CtrTargets;
        TCudaBuffer<const ui32, TMapping> SortedByBinsIndices;
        THolder<THistoryBasedCtrCalcer<TMapping>> HistoryCalcer;
        bool ComputeCatFeatureStatOnFullData = false;
        ui32 Stream = 0;
    };
}
