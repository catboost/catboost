#pragma once

#include "target_func.h"
namespace NCatboostCuda {
    template <class TTarget,
              ETargetType TargetType = TTarget::TargetType()>
    class TPermutationDerCalcer;

    class IPermutationDerCalcer {
    public:
        using TVec = TCudaBuffer<float, NCudaLib::TStripeMapping>;
        using TConstVec = TCudaBuffer<const float, NCudaLib::TStripeMapping>;

        virtual ~IPermutationDerCalcer() {
        }

        virtual void ApproximateAt(const TVec& point,
                                   TVec* value,
                                   TVec* der,
                                   TVec* der2,
                                   ui32 stream = 0) const = 0;

        virtual TConstVec GetWeights(ui32 streamId) const = 0;
    };

    //for pointwise target we could compute derivatives for any permutation of docs and for leaves estimation it's faster to reorder targets
    template <class TTargetFunc>
    class TPermutationDerCalcer<TTargetFunc, ETargetType::Pointwise>: public IPermutationDerCalcer, public TMoveOnly {
    public:
        using TMapping = typename TTargetFunc::TMapping;
        template <class T>
        using TBuffer = TCudaBuffer<T, TMapping>;
        using TVec = TBuffer<float>;
        using TConstVec = TBuffer<const float>;

        TPermutationDerCalcer(TTargetFunc&& target,
                              const TBuffer<const ui32>& indices)
            : Parent(new TTargetFunc(std::move(target)))
        {
            Target = TVec::CopyMapping(indices);
            Gather(Target, Parent->GetTarget().GetTargets(), indices);

            Weights = TVec::CopyMapping(indices);
            Gather(Weights, Parent->GetTarget().GetWeights(), indices);
        }

        TPermutationDerCalcer() = default;

        //point[i] is cursor for document indices[i]
        //der[i] and der2[i] are derivatives for point[i]
        //targets and weights are reordered, so target[i] is target for indices[i]
        void ApproximateAt(const TVec& point,
                           TVec* value,
                           TVec* der,
                           TVec* der2,
                           ui32 stream = 0) const final {
            Parent->Approximate(Target,
                                Weights,
                                point,
                                value,
                                der,
                                der2,
                                stream);
        }

        TConstVec GetWeights(ui32 streamId) const final {
            Y_UNUSED(streamId);
            return Weights.ConstCopyView();
        }

    private:
        THolder<TTargetFunc> Parent;
        TVec Target;
        TVec Weights;
    };

    //der calcer specialization for non-pointwise target (like pairwise/querywise)
    //Querywise targets can't compute permutated derivatives directly (cause we have query grouping)
    template <class TTarget>
    class TPermutationDerCalcer<TTarget, ETargetType::Querywise>: public IPermutationDerCalcer, public TMoveOnly {
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
                           ui32 stream = 0) const final {
            Parent->ApproximateForPermutation(point,
                                              &InverseIndices, /* inverse leaves indices */
                                              value,
                                              der,
                                              der2,
                                              stream);
        }

        TConstVec GetWeights(ui32 streamId) const final {
            TVec tmp;
            tmp.Reset(Indices.GetMapping());
            Gather(tmp, Parent->GetTarget().GetWeights(), Indices, streamId);
            return tmp.ConstCopyView();
        }

    private:
        THolder<TTarget> Parent;
        TBuffer<const ui32> Indices;
        TBuffer<ui32> InverseIndices;
    };

    template <class TTarget>
    inline THolder<IPermutationDerCalcer> CreatePermutationDerCalcer(TTarget&& target,
                                                                     TCudaBuffer<const ui32, typename TTarget::TMapping>&& indices) {
        return new TPermutationDerCalcer<TTarget, TTarget::TargetType()>(std::move(target),
                                                                         std::move(indices));
    }

}
