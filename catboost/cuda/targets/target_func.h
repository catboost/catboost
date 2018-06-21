#pragma once

#include "non_diag_target_der.h"
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/cuda/gpu_data/samples_grouping_gpu.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/libs/options/bootstrap_options.h>
#include <catboost/cuda/cuda_util/gpu_random.h>
#include <catboost/cuda/gpu_data/dataset_base.h>

#define CB_DEFINE_CUDA_TARGET_BUFFERS()       \
    template <class T>                        \
    using TBuffer = TCudaBuffer<T, TMapping>; \
    using TVec = TBuffer<float>;              \
    using TConstVec = TBuffer<const float>;

namespace NCatboostCuda {
    enum ETargetFuncType {
        Pointwise,
        Querywise,
        NonDiagQuerywise
    };

    using TAdditiveStatistic = TMetricHolder;

    inline TAdditiveStatistic MakeSimpleAdditiveStatistic(double sum, double weight) {
        TAdditiveStatistic stat(2);
        stat.Stats[0] = sum;
        stat.Stats[1] = weight;
        return stat;
    }

    /*
     * Target is objective function for samples subset from one dataset
     * Indices are index of samples in dataSet
     * target and weights are gather target/weights for this samples
     */
    template <class TMapping,
              class TDataSet>
    class TTargetFunc: public TMoveOnly {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        //targetFunc constructs are generated on instantiation, so there'll be checking in compile time for support of slices and other mapping-specific stuff
        TTargetFunc(const TDataSet& dataSet,
                    TGpuAwareRandom& random,
                    const TSlice& slice)
            : Target(SliceTarget(dataSet.GetTarget(), slice))
            , DataSet(&dataSet)
            , Random(&random)
        {
        }

        TTargetFunc(const TDataSet& dataSet,
                    TGpuAwareRandom& random)
            : Target(dataSet.GetTarget())
            , DataSet(&dataSet)
            , Random(&random)
        {
        }

        TTargetFunc(const TDataSet& dataSet,
                    TGpuAwareRandom& random,
                    TTarget<TMapping>&& target)
            : Target(std::move(target))
            , DataSet(&dataSet)
            , Random(&random)
        {
        }

        TTargetFunc(const TTargetFunc& target,
                    const TSlice& slice)
            : Target(SliceTarget(target.GetTarget(), slice))
            , DataSet(&target.GetDataSet())
            , Random(target.Random)
        {
        }

        TTargetFunc(const TTargetFunc& target)
            : Target(target.GetTarget())
            , DataSet(&target.GetDataSet())
            , Random(target.Random)
        {
        }

        TTargetFunc(TTargetFunc&& other) = default;

        const TTarget<TMapping>& GetTarget() const {
            return Target;
        }

        const TMapping& GetSamplesMapping() const {
            return Target.GetTargets().GetMapping();
        }

        template <class T>
        TCudaBuffer<T, TMapping> CreateGpuBuffer() const {
            return TCudaBuffer<T, TMapping>::CopyMapping(Target.GetTargets());
        };

        const TDataSet& GetDataSet() const {
            return *DataSet;
        }

        TGpuAwareRandom& GetRandom() const {
            return *Random;
        }

        const TGpuSamplesGrouping<TMapping>& GetSamplesGrouping() const {
            CB_ENSURE(false, "Error, should be unreachable");
        }

        inline double GetTotalWeight() const {
            if (TotalWeight <= 0) {
                auto tmp = TVec::CopyMapping(Target.GetWeights());
                FillBuffer(tmp, 1.0f);
                TotalWeight = DotProduct(tmp, Target.GetWeights());
                if (TotalWeight <= 0) {
                    ythrow yexception() << "Observation weights should be greater or equal zero. Total weight should be greater, than zero";
                }
            }
            return TotalWeight;
        }

    protected:
        TTarget<TMapping> Target;

    private:
        const TDataSet* DataSet;
        TGpuAwareRandom* Random;

        mutable double TotalWeight = 0;
    };

    template <class TMapping,
              class TDataSet>
    class TPointwiseTarget: public TTargetFunc<TMapping, TDataSet> {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TPointwiseTarget(const TDataSet& dataSet,
                         TGpuAwareRandom& random,
                         const TSlice& slice)
            : TTargetFunc<TMapping, TDataSet>(dataSet, random, slice)
        {
        }

        TPointwiseTarget(const TDataSet& dataSet,
                         TGpuAwareRandom& random)
            : TTargetFunc<TMapping, TDataSet>(dataSet, random)
        {
        }

        TPointwiseTarget(const TDataSet& dataSet,
                         TGpuAwareRandom& random,
                         TTarget<TMapping>&& target)
            : TTargetFunc<TMapping, TDataSet>(dataSet,
                                              random,
                                              std::move(target)) {
        }

        TPointwiseTarget(const TPointwiseTarget& target,
                         const TSlice& slice)
            : TTargetFunc<TMapping, TDataSet>(target, slice)
        {
        }

        TPointwiseTarget(const TPointwiseTarget& target)
            : TTargetFunc<TMapping, TDataSet>(target)
        {
        }

        TPointwiseTarget(TPointwiseTarget&& other) = default;

        static constexpr ETargetFuncType TargetType() {
            return ETargetFuncType::Pointwise;
        }
    };

    template <class TMapping,
              class TDataSet>
    class TQuerywiseTarget: public TTargetFunc<TMapping, TDataSet> {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();
        using TParent = TTargetFunc<TMapping, TDataSet>;

        TQuerywiseTarget(const TDataSet& dataSet,
                         TGpuAwareRandom& random,
                         const TSlice& slice)
            : TTargetFunc<TMapping, TDataSet>(dataSet, random, slice)
            , SamplesGrouping(CreateGpuGrouping(dataSet, slice))
        {
        }

        //for template costructs are generated on use. So will fail in compile time with wrong types :)
        TQuerywiseTarget(const TDataSet& dataSet,
                         TGpuAwareRandom& random)
            : TTargetFunc<TMapping, TDataSet>(dataSet, random)
            , SamplesGrouping(CreateGpuGrouping(dataSet))
        {
        }

        //to make stripe target from mirror one
        TQuerywiseTarget(const TQuerywiseTarget<NCudaLib::TMirrorMapping, TDataSet>& basedOn,
                         TTarget<TMapping>&& target)
            : TTargetFunc<TMapping, TDataSet>(basedOn.GetDataSet(),
                                              basedOn.GetRandom(),
                                              std::move(target))
            , SamplesGrouping(MakeStripeGrouping(basedOn.GetSamplesGrouping(),
                                                 TParent::GetTarget().GetIndices())) {
        }

        TQuerywiseTarget(const TQuerywiseTarget& target,
                         const TSlice& slice)
            : TTargetFunc<TMapping, TDataSet>(target, slice)
            , SamplesGrouping(SliceGrouping(target.GetSamplesGrouping(),
                                            slice)) {
        }

        TQuerywiseTarget(const TQuerywiseTarget& target)
            : TTargetFunc<TMapping, TDataSet>(target)
            , SamplesGrouping(target.GetSamplesGrouping().CopyView())
        {
        }

        TQuerywiseTarget(TQuerywiseTarget&& other) = default;

        static constexpr ETargetFuncType TargetType() {
            return ETargetFuncType::Querywise;
        }

        const TGpuSamplesGrouping<TMapping>& GetSamplesGrouping() const {
            return SamplesGrouping;
        }

    private:
        TGpuSamplesGrouping<TMapping> SamplesGrouping;
    };

    template <class TMapping,
              class TDataSet>
    class TNonDiagQuerywiseTarget: public TTargetFunc<TMapping, TDataSet> {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();
        using TParent = TTargetFunc<TMapping, TDataSet>;

        //for template costructs are generated on use. So will fail in compile time with wrong types :)
        TNonDiagQuerywiseTarget(const TDataSet& dataSet,
                                TGpuAwareRandom& random)
            : TTargetFunc<TMapping, TDataSet>(dataSet, random)
            , SamplesGrouping(CreateGpuGrouping(dataSet))
        {
        }

        TNonDiagQuerywiseTarget(TNonDiagQuerywiseTarget&& other) = default;

        static constexpr ETargetFuncType TargetType() {
            return ETargetFuncType::NonDiagQuerywise;
        }

        const TGpuSamplesGrouping<TMapping>& GetSamplesGrouping() const {
            return SamplesGrouping;
        }

    private:
        TGpuSamplesGrouping<TMapping> SamplesGrouping;
    };

    template <template <class TMapping, class> class TTargetFunc,
              class TDataSet>
    inline TTargetFunc<NCudaLib::TStripeMapping, TDataSet> MakeStripeTargetFunc(const TTargetFunc<NCudaLib::TMirrorMapping, TDataSet>& mirrorTarget) {
        const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
        TVector<TSlice> slices(devCount);
        const ui32 docCount = mirrorTarget.GetTarget().GetSamplesMapping().GetObjectsSlice().Size();
        const ui64 docsPerDevice = docCount / devCount;
        const auto& dataSet = mirrorTarget.GetDataSet();
        const IQueriesGrouping& samplesGrouping = dataSet.GetSamplesGrouping();

        ui64 total = 0;

        for (ui32 i = 0; i < devCount; ++i) {
            const ui64 devSize = (i + 1 != devCount ? docsPerDevice : (docCount - total));
            ui64 nextDevDoc = samplesGrouping.NextQueryOffsetForLine(total + devSize - 1);
            slices[i] = TSlice(total, nextDevDoc);
            total = nextDevDoc;
            CB_ENSURE(slices[i].Size(), "Error: insufficient query (or document) count to split data between several GPUs. Can't continue learning");
            CB_ENSURE(slices[i].Right <= docCount);
        }
        NCudaLib::TStripeMapping stripeMapping = NCudaLib::TStripeMapping(std::move(slices));

        return TTargetFunc<NCudaLib::TStripeMapping, TDataSet>(mirrorTarget,
                                                               TTargetHelper<NCudaLib::TMirrorMapping>::StripeView(mirrorTarget.GetTarget(), stripeMapping));
    }

    template <class TTarget>
    inline TTarget TargetSlice(const TTarget& src,
                               const TSlice& slice) {
        return TTarget(src, slice);
    }

    template <class TTargetFunc>
    class TShiftedTargetSlice: public TMoveOnly {
    public:
        using TMapping = typename TTargetFunc::TMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        //for dynamic feature parallel boosting
        TShiftedTargetSlice(const TTargetFunc& target,
                            const TSlice& slice,
                            TConstVec&& sliceShift)
            : Parent(target,
                     slice)
            , Shift(std::move(sliceShift))
        {
            CB_ENSURE(Parent.GetTarget().GetSamplesMapping().GetObjectsSlice() == Shift.GetObjectsSlice());
        }

        //doc parallel stipped objective
        TShiftedTargetSlice(const TTargetFunc& target,
                            TConstVec&& shift)
            : Parent(target)
            , Shift(std::move(shift))
        {
            CB_ENSURE(Parent.GetTarget().GetSamplesMapping().GetObjectsSlice() == Shift.GetObjectsSlice());
        }

        TShiftedTargetSlice(TShiftedTargetSlice&& other) = default;

        void GradientAtZero(TVec& weightedDer,
                            TVec& weights,
                            ui32 stream = 0) const {
            Parent.GradientAt(Shift,
                              weightedDer,
                              weights,
                              stream);
        }

        void NewtonAtZero(TVec& weightedDer,
                          TVec& weightedDer2,
                          ui32 stream = 0) const {
            Parent.NewtonAt(Shift,
                            weightedDer,
                            weightedDer2,
                            stream);
        }

        const TTarget<TMapping>& GetTarget() const {
            return Parent.GetTarget();
        }

        TGpuAwareRandom& GetRandom() const {
            return Parent.GetRandom();
        }

    private:
        TTargetFunc Parent;
        TConstVec Shift;
    };

    template <class TTargetFunc>
    class TPairwiseTargetAtPoint: public TMoveOnly {
    public:
        using TMapping = typename TTargetFunc::TMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        //doc parallel stipped objective
        TPairwiseTargetAtPoint(const TTargetFunc& target,
                               TConstVec&& shift)
            : Parent(target)
            , Shift(std::move(shift))
        {
            CB_ENSURE(Parent.GetTarget().GetSamplesMapping().GetObjectsSlice() == Shift.GetObjectsSlice());
        }

        TPairwiseTargetAtPoint(TPairwiseTargetAtPoint&& other) = default;

        void ComputeStochasticDerivatives(const NCatboostOptions::TBootstrapConfig& bootstrapConfig,
                                          bool isGradient,
                                          TNonDiagQuerywiseTargetDers* result) const {
            if (isGradient) {
                Parent.StochasticGradient(Shift, bootstrapConfig, result);
            } else {
                Parent.StochasticNewton(Shift, bootstrapConfig, result);
            }
        }

        void ComputeDerivatives(TNonDiagQuerywiseTargetDers* result) const {
            Parent.Approximate(Shift, result);
        }

        TGpuAwareRandom& GetRandom() const {
            return Parent.GetRandom();
        }

    private:
        const TTargetFunc& Parent;
        TConstVec Shift;
    };

    template <class TTargetFunc,
              ETargetFuncType FuncType = TTargetFunc::TargetType()>
    class TTargetAtPointTrait {
    public:
        using TConstVec = typename TTargetFunc::TConstVec;
        using Type = TShiftedTargetSlice<TTargetFunc>;

        static Type Create(const TTargetFunc& target,
                           const TSlice& slice,
                           TConstVec&& sliceShift) {
            return Type(target, slice, std::move(sliceShift));
        }

        static Type Create(const TTargetFunc& target,
                           TConstVec&& shift) {
            return Type(target, std::move(shift));
        }
    };

    template <class TTargetFunc>
    class TTargetAtPointTrait<TTargetFunc, ETargetFuncType::NonDiagQuerywise> {
    public:
        using TConstVec = typename TTargetFunc::TConstVec;
        using Type = TPairwiseTargetAtPoint<TTargetFunc>;

        static Type Create(const TTargetFunc& target,
                           TConstVec&& shift) {
            return Type(target,
                        std::move(shift));
        }
    };

}
