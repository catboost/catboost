#pragma once

#include "features_layout.h"

#include <catboost/private/libs/data_types/text.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/system/types.h>
#include <util/generic/noncopyable.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/stream/buffer.h>
#include <util/stream/labeled.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include <climits>
#include <cmath>
#include <type_traits>


namespace NCB {

    //feature values storage optimized for memory usage

    enum class EFeatureValuesType {
        Float,                      //32 bits per feature value
        QuantizedFloat,             //quantized with at most 8 bits (for GPU) or 16 bits (for CPU) per
                                    // feature value.
        HashedCategorical,          //values - 32 bit hashes of original strings
        PerfectHashedCategorical,   //after perfect hashing
        StringText,                 //unoptimized text feature
        TokenizedText,              //32 bits for each token in string
        BinaryPack,                 //aggregate of binary features
        ExclusiveFeatureBundle,     //aggregate of exclusive quantized features
        FeaturesGroup               //aggregate of several quantized float features
    };

    using TFeaturesArraySubsetIndexing = TArraySubsetIndexing<ui32>;
    using TFeaturesArraySubsetInvertedIndexing = TArraySubsetInvertedIndexing<ui32>;
    using TCompressedArraySubset = TArraySubset<TCompressedArray, ui32>;
    using TConstCompressedArraySubset = TArraySubset<const TCompressedArray, ui32>;

    template <class T>
    using TConstPtrArraySubset = TArraySubset<const T*, ui32>;

    class IFeatureValuesHolder: TMoveOnly {
    public:
        virtual ~IFeatureValuesHolder() = default;

        IFeatureValuesHolder(EFeatureValuesType type,
                             ui32 featureId,
                             ui32 size,
                             bool isSparse)
            : Type(type)
            , FeatureId(featureId)
            , Size(size)
            , IsSparse(isSparse)
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
                case EFeatureValuesType::StringText:
                case EFeatureValuesType::TokenizedText:
                    return EFeatureType::Text;
                case EFeatureValuesType::BinaryPack:
                case EFeatureValuesType::ExclusiveFeatureBundle:
                case EFeatureValuesType::FeaturesGroup:
                    CB_ENSURE_INTERNAL(false, "GetFeatureType called for Aggregate type");
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

        bool GetIsSparse() const {
            return IsSparse;
        }

    private:
        EFeatureValuesType Type;
        ui32 FeatureId;
        ui32 Size;
        bool IsSparse;
    };

    using TFeatureColumnPtr = THolder<IFeatureValuesHolder>;


    inline bool IsConsistentWithLayout(
        const IFeatureValuesHolder& feature,
        const TFeaturesLayout& featuresLayout
    ) {
        return featuresLayout.IsCorrectExternalFeatureIdxAndType(feature.GetId(), feature.GetFeatureType());
    }

    /*******************************************************************************************************
     * Common interfaces
     */

    template <class T, EFeatureValuesType TType>
    struct TTypedFeatureValuesHolder : public IFeatureValuesHolder {
        TTypedFeatureValuesHolder(ui32 featureId, ui32 size, bool isSparse)
            : IFeatureValuesHolder(TType, featureId, size, isSparse)
        {}

        virtual TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const = 0;

        virtual IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const = 0;
    };


    template <class T, EFeatureValuesType TType>
    struct TCloneableWithSubsetIndexingValuesHolder : public TTypedFeatureValuesHolder<T, TType>
    {
        TCloneableWithSubsetIndexingValuesHolder(ui32 featureId, ui32 size)
            : TTypedFeatureValuesHolder<T, TType>(featureId, size, /*isSparse*/ false)
        {}

        /* note: subsetIndexing is already a composition - this is an optimization to call compose once per
         * all features data and not for each feature
         */
        virtual THolder<TCloneableWithSubsetIndexingValuesHolder> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const = 0;
    };

    /*******************************************************************************************************
     * Raw data
     */

    template <class T, EFeatureValuesType TType>
    class TPolymorphicArrayValuesHolder: public TCloneableWithSubsetIndexingValuesHolder<T, TType> {
    public:
        TPolymorphicArrayValuesHolder(ui32 featureId,
                                      ITypedArraySubsetPtr<T>&& data)
            : TCloneableWithSubsetIndexingValuesHolder<T, TType>(featureId, data->GetSize())
            , Data(std::move(data))
        {}

        TPolymorphicArrayValuesHolder(ui32 featureId,
                                      TMaybeOwningConstArrayHolder<T> srcData,
                                      const TFeaturesArraySubsetIndexing* subsetIndexing)
            : TPolymorphicArrayValuesHolder(
                featureId,
                MakeIntrusive<TTypeCastArraySubset<T, T>>(std::move(srcData), subsetIndexing)
            )
        {}


        THolder<TCloneableWithSubsetIndexingValuesHolder<T, TType>> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override {
            return MakeHolder<TPolymorphicArrayValuesHolder>(
                this->GetId(),
                Data->CloneWithNewSubsetIndexing(subsetIndexing)
            );
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            TVector<T> result;
            result.yresize(Data->GetSize());
            TArrayRef<T> resultRef = result;

            Data->ParallelForEach(
                [=] (ui32 dstIdx, T value) {
                    resultRef[dstIdx] = value;
                },
                localExecutor
            );
            return TMaybeOwningArrayHolder<T>::CreateOwning(std::move(result));
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            return Data->GetBlockIterator(offset);
        }

        ITypedArraySubsetPtr<T> GetData() const {
            return Data;
        }

    private:
        ITypedArraySubsetPtr<T> Data;
    };

    using TFloatValuesHolder = TTypedFeatureValuesHolder<float, EFeatureValuesType::Float>;
    using TFloatArrayValuesHolder = TPolymorphicArrayValuesHolder<float, EFeatureValuesType::Float>;

    using THashedCatValuesHolder = TTypedFeatureValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;
    using THashedCatArrayValuesHolder = TPolymorphicArrayValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;

    using TStringTextValuesHolder = TTypedFeatureValuesHolder<TString, EFeatureValuesType::StringText>;
    using TStringTextArrayValuesHolder = TPolymorphicArrayValuesHolder<TString, EFeatureValuesType::StringText>;

    /*******************************************************************************************************
     * Quantized/prepared for quantization data
     */

    // calls generic f with 'const T' pointer to raw data of compressedArray with the appropriate T
    template <class F>
    inline void DispatchBitsPerKeyToDataType(
        const TCompressedArray& compressedArray,
        const TStringBuf errorMessagePrefix,
        F&& f
    ) {
        const auto bitsPerKey = compressedArray.GetBitsPerKey();
        const char* rawDataPtr = compressedArray.GetRawPtr();
        switch (bitsPerKey) {
            case 8:
                f((const ui8*)rawDataPtr);
                break;
            case 16:
                f((const ui16*)rawDataPtr);
                break;
            case 32:
                f((const ui32*)rawDataPtr);
                break;
            default:
                CB_ENSURE_INTERNAL(
                    false,
                    errorMessagePrefix << "unsupported bitsPerKey: " << bitsPerKey);
        }
    }


    template <class T, EFeatureValuesType TType>
    class TCompressedValuesHolderImpl : public TCloneableWithSubsetIndexingValuesHolder<T, TType> {
    public:
        using TBase = TCloneableWithSubsetIndexingValuesHolder<T, TType>;

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

        THolder<TCloneableWithSubsetIndexingValuesHolder<T, TType>> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override {
            return MakeHolder<TCompressedValuesHolderImpl>(TBase::GetId(), SrcData, subsetIndexing);
        }

        TConstCompressedArraySubset GetCompressedData() const {
            return {&SrcData, SubsetIndexing};
        }

        template <class T2 = T>
        TConstPtrArraySubset<T2> GetArrayData() const {
            SrcData.CheckIfCanBeInterpretedAsRawArray<T2>();
            return TConstPtrArraySubset<T2>((const T2**)&SrcDataRawPtr, SubsetIndexing);
        }

        // in some cases non-standard T can be useful / more efficient
        template <class T2 = T>
        TMaybeOwningArrayHolder<T2> ExtractValuesT(NPar::TLocalExecutor* localExecutor) const {
            return TMaybeOwningArrayHolder<T2>::CreateOwning(
                ::NCB::GetSubset<T2>(SrcData, *SubsetIndexing, localExecutor)
            );
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            /* TODO(akhropov): implement  TTypedFeatureValuesHolder::GetIterator that will support
             *  non-consecutive data as well. MLTOOLS-3185.
             */

            return SrcData.GetBlockIterator<T>(SubsetIndexing->GetConsecutiveSubsetBeginNonChecked() + offset);
        }

        template <class F>
        void ForEach(F&& f, const NCB::TFeaturesArraySubsetIndexing* featuresSubsetIndexing = nullptr) const {
            if (!featuresSubsetIndexing) {
                featuresSubsetIndexing = SubsetIndexing;
            }
            switch (SrcData.GetBitsPerKey()) {
            case 8:
                NCB::TConstPtrArraySubset<ui8>(
                    GetArrayData<ui8>().GetSrc(),
                    featuresSubsetIndexing
                ).ForEach(std::move(f));
                break;
            case 16:
                NCB::TConstPtrArraySubset<ui16>(
                    GetArrayData<ui16>().GetSrc(),
                    featuresSubsetIndexing
                ).ForEach(std::move(f));
                break;
            case 32:
                NCB::TConstPtrArraySubset<ui32>(
                    GetArrayData<ui32>().GetSrc(),
                    featuresSubsetIndexing
                ).ForEach(std::move(f));
                break;
            default:
                Y_UNREACHABLE();
            }
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            return ExtractValuesT<T>(localExecutor);
        }

        ui32 GetBitsPerKey() const {
            return SrcData.GetBitsPerKey();
        }

    private:
        TCompressedArray SrcData;
        void* SrcDataRawPtr;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
    };

    using IQuantizedFloatValuesHolder = TTypedFeatureValuesHolder<ui8, EFeatureValuesType::QuantizedFloat>;

    using ICloneableQuantizedFloatValuesHolder
        = TCloneableWithSubsetIndexingValuesHolder<ui8, EFeatureValuesType::QuantizedFloat>;

    using TQuantizedFloatValuesHolder
        = TCompressedValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;

    using IQuantizedCatValuesHolder
        = TTypedFeatureValuesHolder<ui32, EFeatureValuesType::PerfectHashedCategorical>;

    using ICloneableQuantizedCatValuesHolder
        = TCloneableWithSubsetIndexingValuesHolder<ui32, EFeatureValuesType::PerfectHashedCategorical>;

    using TTokenizedTextValuesHolder = TTypedFeatureValuesHolder<TText, EFeatureValuesType::TokenizedText>;
    using TTokenizedTextArrayValuesHolder = TPolymorphicArrayValuesHolder<TText, EFeatureValuesType::TokenizedText>;

}
