#pragma once

#include "features_layout.h"
#include "packed_binary_features.h"

#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/system/types.h>
#include <util/generic/noncopyable.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/stream/buffer.h>
#include <util/system/yassert.h>

#include <climits>
#include <cmath>
#include <type_traits>


namespace NCB {

    //feature values storage optimized for memory usage

    enum class EFeatureValuesType {
        Float,                      //32 bits per feature value
        QuantizedFloat,             //at most 8 bits per feature value. Contains grid
        HashedCategorical,          //values - 32 bit hashes of original strings
        PerfectHashedCategorical,   //after perfect hashing.
    };

    using TFeaturesArraySubsetIndexing = TArraySubsetIndexing<ui32>;
    using TCompressedArraySubset = TArraySubset<TCompressedArray, ui32>;
    using TConstCompressedArraySubset = TArraySubset<const TCompressedArray, ui32>;

    template <class T>
    using TConstPtrArraySubset = TArraySubset<const T*, ui32>;

    class IFeatureValuesHolder: TMoveOnly {
    public:
        virtual ~IFeatureValuesHolder() = default;

        IFeatureValuesHolder(EFeatureValuesType type,
                             ui32 featureId,
                             ui32 size)
            : Type(type)
            , FeatureId(featureId)
            , Size(size)
        {
        }

        IFeatureValuesHolder(IFeatureValuesHolder&& arg) noexcept = default;
        IFeatureValuesHolder& operator=(IFeatureValuesHolder&& arg) noexcept = default;

        EFeatureType GetFeatureType() const {
            switch (Type) {
                case EFeatureValuesType::Float:
                case EFeatureValuesType::QuantizedFloat:
                    return EFeatureType::Float;
                case EFeatureValuesType::HashedCategorical:
                case EFeatureValuesType::PerfectHashedCategorical:
                    return EFeatureType::Categorical;
            }
            Y_FAIL("This place should be inaccessible");
            return EFeatureType::Float; // to keep compiler happy
        }

        EFeatureValuesType GetType() const {
            return Type;
        }

        ui32 GetSize() const {
            return Size;
        }

        ui32 GetId() const {
            return FeatureId;
        }

    private:
        EFeatureValuesType Type;
        ui32 FeatureId;
        ui32 Size;
    };

    using TFeatureColumnPtr = THolder<IFeatureValuesHolder>;


    inline bool IsConsistentWithLayout(
        const IFeatureValuesHolder& feature,
        const TFeaturesLayout& featuresLayout
    ) {
        return featuresLayout.IsCorrectExternalFeatureIdxAndType(feature.GetId(), feature.GetFeatureType());
    }


    /*******************************************************************************************************
     * Raw data
     */

    template <class T, EFeatureValuesType TType>
    class TArrayValuesHolder: public IFeatureValuesHolder {
    public:
        TArrayValuesHolder(ui32 featureId,
                           TMaybeOwningConstArrayHolder<T> srcData,
                           const TFeaturesArraySubsetIndexing* subsetIndexing)
            : IFeatureValuesHolder(TType,
                                   featureId,
                                   subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        const TMaybeOwningConstArraySubset<T, ui32> GetArrayData() const {
            return {&SrcData, SubsetIndexing};
        }

    private:
        TMaybeOwningConstArrayHolder<T> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
    };

    using TFloatValuesHolder = TArrayValuesHolder<float, EFeatureValuesType::Float>;

    using THashedCatValuesHolder = TArrayValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;


    /*******************************************************************************************************
     * Quantized/prepared for quantization data
     */

    template <class TBase>
    class TCompressedValuesHolderImpl : public TBase {
    public:
        TCompressedValuesHolderImpl(ui32 featureId,
                                    TCompressedArray srcData,
                                    const TFeaturesArraySubsetIndexing* subsetIndexing)
            : TBase(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SrcDataRawPtr(SrcData.GetRawPtr())
            , SubsetIndexing(subsetIndexing)
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<TBase> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override {
            return MakeHolder<TCompressedValuesHolderImpl>(TBase::GetId(), SrcData, subsetIndexing);
        }

        TConstCompressedArraySubset GetCompressedData() const {
            return {&SrcData, SubsetIndexing};
        }

        template <class T = typename TBase::TValueType>
        TConstPtrArraySubset<T> GetArrayData() const {
            SrcData.CheckIfCanBeInterpretedAsRawArray<T>();
            return TConstPtrArraySubset<T>((const T**)&SrcDataRawPtr, SubsetIndexing);
        }

        // in some cases non-standard T can be useful / more efficient
        template <class T = typename TBase::TValueType>
        TMaybeOwningArrayHolder<T> ExtractValuesT(NPar::TLocalExecutor* localExecutor) const {
            return TMaybeOwningArrayHolder<T>::CreateOwning(
                ::NCB::GetSubset<T>(SrcData, *SubsetIndexing, localExecutor)
            );
        }

        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValues(
            NPar::TLocalExecutor* localExecutor
        ) const override {
            return ExtractValuesT<typename TBase::TValueType>(localExecutor);
        }

        ui32 GetBitsPerKey() const {
            return SrcData.GetBitsPerKey();
        }

    private:
        TCompressedArray SrcData;
        void* SrcDataRawPtr;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
    };

    template <class TBase>
    class TPackedBinaryValuesHolderImpl : public TBase {
    public:
        TPackedBinaryValuesHolderImpl(ui32 featureId,
                                      NCB::TMaybeOwningArrayHolder<NCB::TBinaryFeaturesPack> srcData,
                                      ui8 bitIdx,
                                      const TFeaturesArraySubsetIndexing* subsetIndexing)
            : TBase(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , BitIdx(bitIdx)
            , SubsetIndexing(subsetIndexing)
        {
            CB_ENSURE(
                BitIdx < sizeof(NCB::TBinaryFeaturesPack) * CHAR_BIT,
                "BitIdx=" << BitIdx << " is bigger than limit ("
                << sizeof(NCB::TBinaryFeaturesPack) * CHAR_BIT << ')'
            );
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        THolder<TBase> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override {
            return MakeHolder<TPackedBinaryValuesHolderImpl>(
                TBase::GetId(),
                SrcData,
                BitIdx,
                subsetIndexing
            );
        }

        // in some cases non-standard T can be useful / more efficient
        template <class T = typename TBase::TValueType>
        TMaybeOwningArrayHolder<T> ExtractValuesT(NPar::TLocalExecutor* localExecutor) const {
            TConstArrayRef<NCB::TBinaryFeaturesPack> srcData = *SrcData;
            NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << BitIdx;

            TVector<T> dst;
            dst.yresize(SubsetIndexing->Size());

            SubsetIndexing->ParallelForEach(
                [&dst, srcData, bitIdx = BitIdx, bitMask] (ui32 idx, ui32 srcDataIdx) {
                    dst[idx] = (srcData[srcDataIdx] & bitMask) >> bitIdx;
                },
                localExecutor
            );

            return TMaybeOwningArrayHolder<T>::CreateOwning(std::move(dst));
        }

        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValues(
            NPar::TLocalExecutor* localExecutor
        ) const override {
            return ExtractValuesT<typename TBase::TValueType>(localExecutor);
        }

        ui8 GetBitIdx() const {
            return BitIdx;
        }

    private:
        NCB::TMaybeOwningArrayHolder<NCB::TBinaryFeaturesPack> SrcData;
        ui8 BitIdx;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
    };


    /* interface instead of concrete TQuantizedFloatValuesHolder because there is
     * an alternative implementation TExternalFloatValuesHolder for GPU
     */
    class IQuantizedFloatValuesHolder: public IFeatureValuesHolder {
    public:
        using TValueType = ui8;
    public:
        IQuantizedFloatValuesHolder(const ui32 featureId,
                                    ui32 size)
            : IFeatureValuesHolder(EFeatureValuesType::QuantizedFloat,
                                   featureId,
                                   size)
        {}

        /* note: subsetIndexing is already a composition - this is an optimization to call compose once per
         * all features data and not for each feature
         */
        virtual THolder<IQuantizedFloatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const = 0;

        /* For one-time use on GPU.
         * On CPU TQuantizedCatValuesHolder::GetArrayData should be used
         */
        virtual TMaybeOwningArrayHolder<ui8> ExtractValues(
            NPar::TLocalExecutor* localExecutor
        ) const = 0;
    };

    using TQuantizedFloatValuesHolder = TCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>;
    using TQuantizedFloatPackedBinaryValuesHolder = TPackedBinaryValuesHolderImpl<IQuantizedFloatValuesHolder>;

    /* interface instead of concrete TQuantizedFloatValuesHolder because there is
     * an alternative implementation TExternalFloatValuesHolder for GPU
     */
    class IQuantizedCatValuesHolder: public IFeatureValuesHolder {
    public:
        using TValueType = ui32;
    public:
        IQuantizedCatValuesHolder(const ui32 featureId,
                                  ui32 size)
            : IFeatureValuesHolder(EFeatureValuesType::PerfectHashedCategorical,
                                   featureId,
                                   size)
        {}

        /* note: subsetIndexing is already a composition - this is an optimization to call compose once per
         * all features data and not for each feature
         */
        virtual THolder<IQuantizedCatValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const = 0;

        /* For one-time use on GPU.
         * On CPU TQuantizedCatValuesHolder::GetArrayData should be used
         */
        virtual TMaybeOwningArrayHolder<ui32> ExtractValues(
            NPar::TLocalExecutor* localExecutor
        ) const = 0;
    };

    using TQuantizedCatValuesHolder = TCompressedValuesHolderImpl<IQuantizedCatValuesHolder>;
    using TQuantizedCatPackedBinaryValuesHolder = TPackedBinaryValuesHolderImpl<IQuantizedCatValuesHolder>;

}
