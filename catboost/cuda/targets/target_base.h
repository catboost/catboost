#pragma once

#include "quality_metric_helpers.h"
#include <catboost/cuda/gpu_data/samples_grouping_gpu.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>

#define CB_DEFINE_CUDA_TARGET_BUFFERS()       \
    template <class T>                        \
    using TBuffer = TCudaBuffer<T, TMapping>; \
    using TVec = TBuffer<float>;              \
    using TConstVec = TBuffer<const float>;

namespace NCatboostCuda {
    enum ETargetType {
        Pointwise,
        Querywise
    };

    template <class TMapping,
              class TDataSet>
    class TTargetBase: public TMoveOnly {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TTargetBase(const TDataSet& dataSet,
                    TRandom& random,
                    const TSlice& slice)
            : Target(dataSet.GetTarget().SliceView(slice))
            , Weights(dataSet.GetWeights().SliceView(slice))
            , DataSet(&dataSet)
            , Indices(dataSet.GetIndices().SliceView(slice))
            , Random(&random)
        {
        }

        TTargetBase(const TDataSet& dataSet,
                    TRandom& random,
                    TCudaBuffer<const float, TMapping>&& target,
                    TCudaBuffer<const float, TMapping>&& weights,
                    TCudaBuffer<const ui32, TMapping>&& indices)
            : Target(std::move(target))
            , Weights(std::move(weights))
            , DataSet(&dataSet)
            , Indices(std::move(indices))
            , Random(&random)
        {
        }

        TTargetBase(const TTargetBase& target,
                    const TSlice& slice)
            : Target(target.GetTarget().SliceView(slice))
            , Weights(target.GetWeights().SliceView(slice))
            , DataSet(&target.GetDataSet())
            , Indices(target.GetIndices().SliceView(slice))
            , Random(target.Random)
        {
        }

        TTargetBase(TTargetBase&& other) = default;

        const TConstVec& GetTarget() const {
            return Target;
        }

        const TConstVec& GetWeights() const {
            return Weights;
        }

        const TBuffer<const ui32>& GetIndices() const {
            return Indices;
        }

        template <class T>
        TCudaBuffer<T, TMapping> CreateGpuBuffer() const {
            return TCudaBuffer<T, TMapping>::CopyMapping(Target);
        };

        const TDataSet& GetDataSet() const {
            return *DataSet;
        }

        TRandom& GetRandom() const {
            return *Random;
        }

        inline double GetTotalWeight() const {
            if (TotalWeight <= 0) {
                auto tmp = TVec::CopyMapping(Weights);
                FillBuffer(tmp, 1.0f);
                TotalWeight = DotProduct(tmp, Weights);
                if (TotalWeight <= 0) {
                    ythrow yexception() << "Observation weights should be greater or equal zero. Total weight should be greater, than zero";
                }
            }
            return TotalWeight;
        }

    protected:
        TConstVec Target;
        TConstVec Weights;

    private:
        const TDataSet* DataSet;
        TBuffer<const ui32> Indices;
        TRandom* Random;

        mutable double TotalWeight = 0;
    };

    template <class TMapping,
              class TDataSet>
    class TPointwiseTarget: public TTargetBase<TMapping, TDataSet> {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TPointwiseTarget(const TDataSet& dataSet,
                         TRandom& random,
                         const TSlice& slice)
            : TTargetBase<TMapping, TDataSet>(dataSet, random, slice)
        {
        }

        TPointwiseTarget(const TDataSet& dataSet,
                         TRandom& random,
                         TCudaBuffer<const float, TMapping>&& target,
                         TCudaBuffer<const float, TMapping>&& weights,
                         TCudaBuffer<const ui32, TMapping>&& indices)
            : TTargetBase<TMapping, TDataSet>(dataSet, random, std::move(target), std::move(weights), std::move(indices))
        {
        }

        TPointwiseTarget(const TPointwiseTarget& target,
                         const TSlice& slice)
            : TTargetBase<TMapping, TDataSet>(target, slice)
        {
        }

        TPointwiseTarget(TPointwiseTarget&& other) = default;

        static constexpr ETargetType TargetType() {
            return ETargetType::Pointwise;
        }
    };

    template <class TMapping,
              class TDataSet>
    class TQuerywiseTarget: public TTargetBase<TMapping, TDataSet> {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();
        using TParent = TTargetBase<TMapping, TDataSet>;

        TQuerywiseTarget(const TDataSet& dataSet,
                         TRandom& random,
                         const TSlice& slice)
            : TTargetBase<TMapping, TDataSet>(dataSet, random, slice)
            , SamplesGrouping(CreateGpuGrouping(dataSet, slice))
        {
        }

        //to make stripe target from mirror one
        TQuerywiseTarget(const TQuerywiseTarget<NCudaLib::TMirrorMapping, TDataSet>& basedOn,
                         TCudaBuffer<const float, NCudaLib::TStripeMapping>&& target,
                         TCudaBuffer<const float, NCudaLib::TStripeMapping>&& weights,
                         TCudaBuffer<const ui32, NCudaLib::TStripeMapping>&& indices)
            : TTargetBase<TMapping, TDataSet>(basedOn.GetDataSet(), basedOn.GetRandom(),
                                              std::move(target), std::move(weights), std::move(indices))
            , SamplesGrouping(MakeStripeGrouping(basedOn.GetSamplesGrouping(), TParent::GetIndices()))
        {
        }

        TQuerywiseTarget(const TQuerywiseTarget& target,
                         const TSlice& slice)
            : TTargetBase<TMapping, TDataSet>(target, slice)
            , SamplesGrouping(SliceGrouping(target.GetSamplesGrouping(), slice))
        {
        }

        TQuerywiseTarget(TQuerywiseTarget&& other) = default;

        static constexpr ETargetType TargetType() {
            return ETargetType::Querywise;
        }

        const TGpuSamplesGrouping<TMapping>& GetSamplesGrouping() const {
            return SamplesGrouping;
        }

    private:
        TGpuSamplesGrouping<TMapping> SamplesGrouping;
    };

    template <template <class TMapping, class> class TTarget, class TDataSet>
    inline TTarget<NCudaLib::TStripeMapping, TDataSet> MakeStripeTarget(const TTarget<NCudaLib::TMirrorMapping, TDataSet>& mirrorTarget) {
        const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
        TVector<TSlice> slices(devCount);
        const ui32 docCount = mirrorTarget.GetTarget().GetObjectsSlice().Size();
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

        return TTarget<NCudaLib::TStripeMapping, TDataSet>(mirrorTarget,
                                                           NCudaLib::StripeView(mirrorTarget.GetTarget(),
                                                                                stripeMapping),
                                                           NCudaLib::StripeView(mirrorTarget.GetWeights(),
                                                                                stripeMapping),
                                                           NCudaLib::StripeView(mirrorTarget.GetIndices(),
                                                                                stripeMapping));
    }

    template <class TTarget>
    inline TTarget TargetSlice(const TTarget& src,
                               const TSlice& slice) {
        return TTarget(src, slice);
    }

    template <class TTarget,
              ETargetType TargetType = TTarget::TargetType()>
    class TPermutationDerCalcer;

    //for pointwise target we could compute derivatives for any permutation of docs and for leaves estimation it's faster to reorder targets
    template <class TTarget>
    class TPermutationDerCalcer<TTarget, ETargetType::Pointwise>: public TMoveOnly {
    public:
        using TMapping = typename TTarget::TMapping;
        template <class T>
        using TBuffer = TCudaBuffer<T, TMapping>;
        using TVec = TBuffer<float>;
        using TConstVec = TBuffer<const float>;

        TPermutationDerCalcer(TTarget&& target,
                              const TBuffer<const ui32>& indices)
            : Parent(new TTarget(std::move(target)))
        {
            Target = TVec::CopyMapping(indices);
            Gather(Target, Parent->GetTarget(), indices);

            Weights = TVec::CopyMapping(indices);
            Gather(Weights, Parent->GetWeights(), indices);
        }

        TPermutationDerCalcer() = default;

        //point[i] is cursor for document indices[i]
        //der[i] and der2[i] are derivatives for point[i]
        //targets and weights are reordered, so target[i] is target for indices[i]
        void ApproximateAt(const TVec& point,
                           TVec* value,
                           TVec* der,
                           TVec* der2,
                           ui32 stream = 0) const {
            Parent->Approximate(Target,
                                Weights,
                                point,
                                value,
                                der,
                                der2,
                                stream);
        }

        TConstVec GetWeights(ui32 streamId) const {
            Y_UNUSED(streamId);
            return Weights.ConstCopyView();
        }

    private:
        THolder<TTarget> Parent;
        TVec Target;
        TVec Weights;
    };

    //der calcer specialization for non-pointwise target (like pairwise/querywise)
    //Querywise targets can't compute permutated derivatives directly (cause we have query grouping)
    template <class TTarget>
    class TPermutationDerCalcer<TTarget, ETargetType::Querywise>: public TMoveOnly {
    public:
        using TMapping = typename TTarget::TMapping;
        template <class T>
        using TBuffer = TCudaBuffer<T, TMapping>;
        using TVec = TBuffer<float>;

        TPermutationDerCalcer(TTarget&& target,
                              TBuffer<const ui32>&& indices)
            : Parent(new TTarget(std::move(target)))
        {
            Indices = std::move(indices);
            InverseIndices.Reset(Indices.GetMapping());
            InversePermutation(Indices, InverseIndices);
        }

        TPermutationDerCalcer() = default;

        void ApproximateAt(const TVec& point,
                           TVec* value,
                           TVec* der,
                           TVec* der2,
                           ui32 stream = 0) const {
            Parent->ApproximateForPermutation(point,
                                              &InverseIndices, /* inverse leaves indices */
                                              value,
                                              der,
                                              der2,
                                              stream);
        }

        TVec GetWeights(ui32 streamId) const {
            TVec tmp;
            tmp.Reset(Indices.GetMapping());
            Gather(tmp, Parent->GetWeights(), Indices, streamId);
            return tmp;
        }

    private:
        THolder<TTarget> Parent;
        TBuffer<const ui32> Indices;
        TBuffer<ui32> InverseIndices;
    };

    template <class TTarget>
    inline TPermutationDerCalcer<TTarget> CreateDerCalcer(TTarget&& target,
                                                          TCudaBuffer<const ui32, typename TTarget::TMapping>&& indices) {
        return TPermutationDerCalcer<TTarget, TTarget::TargetType()>(std::move(target),
                                                                     std::move(indices));
    }

    template <class TTarget>
    class TShiftedTargetSlice: public TMoveOnly {
    public:
        using TMapping = typename TTarget::TMapping;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TShiftedTargetSlice(const TTarget& target,
                            const TSlice& slice,
                            TConstVec&& sliceShift)
            : Parent(target, slice)
            , Shift(std::move(sliceShift))
        {
            CB_ENSURE(Parent.GetTarget().GetObjectsSlice() == sliceShift.GetObjectsSlice());
        }

        TShiftedTargetSlice(TShiftedTargetSlice&& other) = default;

        void GradientAtZero(TVec& der,
                            TVec& weights,
                            ui32 stream = 0) const {
            Parent.GradientAt(Shift, der, weights, stream);
        }

        const TConstVec& GetTarget() const {
            return Parent.GetTarget();
        }

        const TBuffer<const ui32>& GetIndices() const {
            return Parent.GetIndices();
        }

        const TConstVec& GetWeights() const {
            return Parent.GetWeights();
        }

        TRandom& GetRandom() const {
            return Parent.GetRandom();
        }

    private:
        TTarget Parent;
        TConstVec Shift;
    };
}
