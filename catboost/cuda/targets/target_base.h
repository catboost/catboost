#pragma once

#include "quality_metric_helpers.h"
#include "target_options.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_util/cpu_random.h>
#include <catboost/cuda/gpu_data/fold_based_dataset.h>

#define CB_DEFINE_CUDA_TARGET_BUFFERS()       \
    template <class T>                        \
    using TBuffer = TCudaBuffer<T, TMapping>; \
    using TVec = TBuffer<float>;              \
    using TConstVec = TBuffer<const float>;

namespace NCatboostCuda
{
    template<class TMapping,
            class TDataSet>
    class TPointwiseTarget: public TMoveOnly
    {
    public:
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        ~TPointwiseTarget()
        {
        }

        TPointwiseTarget(const TDataSet& dataSet,
                         TRandom& random,
                         const TSlice& slice,
                         const TTargetOptions& targetOptions)
                : DataSet(&dataSet)
                  , TargetOptions(&targetOptions)
                  , Target(dataSet.GetTarget().SliceView(slice))
                  , Weights(dataSet.GetWeights().SliceView(slice))
                  , Indices(dataSet.GetIndices().SliceView(slice))
                  , Random(&random)
        {
        }

        TPointwiseTarget(const TDataSet& dataSet,
                         TRandom& random,
                         TCudaBuffer<const float, TMapping>&& target,
                         TCudaBuffer<const float, TMapping>&& weights,
                         TCudaBuffer<const ui32, TMapping>&& indices,
                         const TTargetOptions& targetOptions)
                : DataSet(&dataSet)
                  , TargetOptions(&targetOptions)
                  , Target(std::move(target))
                  , Weights(std::move(weights))
                  , Indices(std::move(indices))
                  , Random(&random)
        {
        }

        TPointwiseTarget(const TPointwiseTarget& target,
                         const TSlice& slice)
                : DataSet(&target.GetDataSet())
                  , TargetOptions(&target.GetTargetOptions())
                  , Target(target.GetTarget().SliceView(slice))
                  , Weights(target.GetWeights().SliceView(slice))
                  , Indices(target.GetIndices().SliceView(slice))
                  , Random(target.Random)
        {
        }

        TPointwiseTarget(TPointwiseTarget&& other) = default;

        const TConstVec& GetTarget() const
        {
            return Target;
        }

        const TConstVec& GetWeights() const
        {
            return Weights;
        }

        const TBuffer<const ui32>& GetIndices() const
        {
            return Indices;
        }

        const TTargetOptions& GetTargetOptions() const
        {
            return *TargetOptions;
        }

        template<class T>
        TCudaBuffer<T, TMapping> CreateGpuBuffer() const
        {
            return TCudaBuffer<T, TMapping>::CopyMapping(Target);
        };

        const TDataSet& GetDataSet() const
        {
            return *DataSet;
        }

        TRandom& GetRandom() const
        {
            return *Random;
        }

        inline double GetTotalWeight() const
        {
            if (TotalWeight <= 0)
            {
                auto tmp = TVec::CopyMapping(Weights);
                FillBuffer(tmp, 1.0f);
                TotalWeight = DotProduct(tmp, Weights);
                if (TotalWeight <= 0)
                {
                    ythrow yexception()
                            << "Observation weights should be greater or equal zero. Total weight should be greater, than zero";
                }
            }
            return TotalWeight;
        }

    private:
        const TDataSet* DataSet;
        const TTargetOptions* TargetOptions;
        TConstVec Target;
        TConstVec Weights;
        TBuffer<const ui32> Indices;
        TRandom* Random;

        mutable double TotalWeight = 0;
    };

    template<template<class TMapping, class> class TTarget, class TDataSet>
    inline TTarget<NCudaLib::TStripeMapping, TDataSet>
    MakeStripeTarget(const TTarget<NCudaLib::TMirrorMapping, TDataSet>& mirrorTarget)
    {
        NCudaLib::TStripeMapping stripeMapping = NCudaLib::TStripeMapping::SplitBetweenDevices(
                mirrorTarget.GetIndices().GetObjectsSlice().Size());

        return TTarget<NCudaLib::TStripeMapping, TDataSet>(mirrorTarget.GetDataSet(),
                                                           mirrorTarget.GetRandom(),
                                                           NCudaLib::StripeView(mirrorTarget.GetTarget(),
                                                                                stripeMapping),
                                                           NCudaLib::StripeView(mirrorTarget.GetWeights(),
                                                                                stripeMapping),
                                                           NCudaLib::StripeView(mirrorTarget.GetIndices(),
                                                                                stripeMapping),
                                                           mirrorTarget.GetTargetOptions());
    }

    template<class TTarget>
    inline TTarget TargetSlice(const TTarget& src,
                               const TSlice& slice)
    {
        return TTarget(src, slice);
    }

    template<class TTarget>
    class TPermutationDerCalcer: public TMoveOnly
    {
    public:
        using TMapping = typename TTarget::TMapping;
        template<class T>
        using TBuffer = TCudaBuffer<T, TMapping>;
        using TVec = TBuffer<float>;

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

        void ApproximateAt(const TVec& point,
                           TVec* value,
                           TVec* der,
                           TVec* der2,
                           ui32 stream = 0
        ) const
        {
            Parent->Approximate(Target,
                                Weights,
                                point,
                                value,
                                der,
                                der2,
                                stream);
        }

        const TVec& GetWeights() const
        {
            return Weights;
        }

    private:
        THolder<TTarget> Parent;
        TVec Target;
        TVec Weights;
    };

    template<class TTarget>
    inline TPermutationDerCalcer<TTarget> CreateDerCalcer(TTarget&& target,
                                                          const TCudaBuffer<const ui32, typename TTarget::TMapping>& indices)
    {
        return TPermutationDerCalcer<TTarget>(std::move(target), indices);
    }

    template<class TTarget>
    class TShiftedTargetSlice: public TMoveOnly
    {
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

        void GradientAtZero(TVec& der, ui32 stream = 0) const
        {
            Parent.GradientAt(Shift, der, stream);
        }

        const TConstVec& GetTarget() const
        {
            return Parent.GetTarget();
        }

        const TBuffer<const ui32>& GetIndices() const
        {
            return Parent.GetIndices();
        }

        const TConstVec& GetWeights() const
        {
            return Parent.GetWeights();
        }

        TRandom& GetRandom() const
        {
            return Parent.GetRandom();
        }

    private:
        TTarget Parent;
        TConstVec Shift;
    };
}

