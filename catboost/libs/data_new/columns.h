#pragma once

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

    using TFeaturesArraySubsetIndexing = TArraySubsetIndexing<ui64>;
    using TCompressedArraySubset = TArraySubset<TCompressedArray, ui64>;

    template <class T>
    using TConstPtrArraySubset = TArraySubset<const T*, ui64>;

    class IFeatureValuesHolder: TMoveOnly {
    public:
        virtual ~IFeatureValuesHolder() = default;

        IFeatureValuesHolder(EFeatureValuesType type,
                             ui32 featureId,
                             ui64 size)
            : Type(type)
            , FeatureId(featureId)
            , Size(size)
        {
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
        ui64 Size;
    };

    using TFeatureColumnPtr = THolder<IFeatureValuesHolder>;


    /*******************************************************************************************************
     * Raw data
     */

    template <class T, EFeatureValuesType TType>
    class TArrayValuesHolder: public IFeatureValuesHolder {
    public:
        TArrayValuesHolder(ui32 featureId,
                           TMaybeOwningArrayHolder<T> srcData,
                           const TFeaturesArraySubsetIndexing* subsetIndexing)
            : IFeatureValuesHolder(TType,
                                   featureId,
                                   subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , Data(&SrcData, subsetIndexing)
        {}

        const TMaybeOwningArraySubset<T>& GetArrayData() const {
            return Data;
        }

    private:
        TMaybeOwningArrayHolder<T> SrcData;
        TMaybeOwningArraySubset<T> Data;
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
            , Data(&SrcData, subsetIndexing)
        {}

        const TCompressedArraySubset& GetCompressedData() const {
            return Data;
        }

        template <class T = typename TBase::TValueType>
        TConstPtrArraySubset<T> GetArrayData() const {
            SrcData.CheckIfCanBeInterpretedAsRawArray<T>();
            return TConstPtrArraySubset<T>((const T**)&SrcDataRawPtr, Data.GetSubsetIndexing());
        }

        // in some cases non-standard T can be useful / more efficient
        template <class T = typename TBase::TValueType>
        TMaybeOwningArrayHolder<T> ExtractValuesT(NPar::TLocalExecutor* localExecutor) const {
            return ParallelExtractValues<T>(Data, localExecutor);
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
        TCompressedArraySubset Data;
    };


    /* interface instead of concrete TQuantizedFloatValuesHolder because there is
     * an alternative implementation TExternalFloatValuesHolder for GPU
     */
    class IQuantizedFloatValuesHolder: public IFeatureValuesHolder {
    public:
        using TValueType = ui8;
    public:
        IQuantizedFloatValuesHolder(const ui32 featureId,
                                    ui64 size)
            : IFeatureValuesHolder(EFeatureValuesType::QuantizedFloat,
                                   featureId,
                                   size)
        {}

        /* For one-time use on GPU.
         * On CPU TQuantizedCatValuesHolder::GetArrayData should be used
         */
        virtual TMaybeOwningArrayHolder<ui8> ExtractValues(
            NPar::TLocalExecutor* localExecutor
        ) const = 0;
    };

    using TQuantizedFloatValuesHolder = TCompressedValuesHolderImpl<IQuantizedFloatValuesHolder>;


    /* interface instead of concrete TQuantizedFloatValuesHolder because there is
         * an alternative implementation TExternalFloatValuesHolder for GPU
         */
    class IQuantizedCatValuesHolder: public IFeatureValuesHolder {
    public:
        using TValueType = ui32;
    public:
        IQuantizedCatValuesHolder(const ui32 featureId,
                                  ui64 size)
            : IFeatureValuesHolder(EFeatureValuesType::PerfectHashedCategorical,
                                   featureId,
                                   size)
        {}

        /* For one-time use on GPU.
         * On CPU TQuantizedCatValuesHolder::GetArrayData should be used
         */
        virtual TMaybeOwningArrayHolder<ui32> ExtractValues(
            NPar::TLocalExecutor* localExecutor
        ) const = 0;
    };

    using TQuantizedCatValuesHolder = TCompressedValuesHolderImpl<IQuantizedCatValuesHolder>;

}
