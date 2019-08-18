#pragma once

#include "exclusive_feature_bundling.h"
#include "feature_grouping.h"
#include "features_layout.h"
#include "packed_binary_features.h"

#include <catboost/libs/data_types/text.h>
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
#include <util/stream/labeled.h>
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
     * Common interfaces
     */

    template <class T, EFeatureValuesType TType>
    struct TTypedFeatureValuesHolder : public IFeatureValuesHolder {
        TTypedFeatureValuesHolder(ui32 featureId, ui32 size)
            : IFeatureValuesHolder(TType, featureId, size)
        {}

        virtual TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const = 0;
    };


    template <class T, EFeatureValuesType TType>
    struct TCloneableWithSubsetIndexingValuesHolder : public TTypedFeatureValuesHolder<T, TType>
    {
        TCloneableWithSubsetIndexingValuesHolder(ui32 featureId, ui32 size)
            : TTypedFeatureValuesHolder<T, TType>(featureId, size)
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
    class TArrayValuesHolder: public TCloneableWithSubsetIndexingValuesHolder<T, TType> {
    public:
        TArrayValuesHolder(ui32 featureId,
                           TMaybeOwningConstArrayHolder<T> srcData,
                           const TFeaturesArraySubsetIndexing* subsetIndexing)
            : TCloneableWithSubsetIndexingValuesHolder<T, TType>(featureId, subsetIndexing->Size())
            , SrcData(std::move(srcData))
            , SubsetIndexing(subsetIndexing)
        {
            CB_ENSURE(SubsetIndexing, "subsetIndexing is empty");
        }

        const TMaybeOwningConstArraySubset<T, ui32> GetArrayData() const {
            return {&SrcData, SubsetIndexing};
        }

        THolder<TCloneableWithSubsetIndexingValuesHolder<T, TType>> CloneWithNewSubsetIndexing(
            const TFeaturesArraySubsetIndexing* subsetIndexing
        ) const override {
            return MakeHolder<TArrayValuesHolder>(this->GetId(), SrcData, subsetIndexing);
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            return TMaybeOwningArrayHolder<T>::CreateOwning(
                GetSubset<T>(*SrcData, *SubsetIndexing, localExecutor)
            );
        }

    private:
        TMaybeOwningConstArrayHolder<T> SrcData;
        const TFeaturesArraySubsetIndexing* SubsetIndexing;
    };


    using TFloatValuesHolder = TTypedFeatureValuesHolder<float, EFeatureValuesType::Float>;
    using TFloatArrayValuesHolder = TArrayValuesHolder<float, EFeatureValuesType::Float>;

    using THashedCatValuesHolder = TTypedFeatureValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;
    using THashedCatArrayValuesHolder = TArrayValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;

    using TStringTextValuesHolder = TTypedFeatureValuesHolder<TString, EFeatureValuesType::StringText>;
    using TStringTextArrayValuesHolder = TArrayValuesHolder<TString, EFeatureValuesType::StringText>;


    /*******************************************************************************************************
     * Quantized/prepared for quantization data
     */

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

    using TBinaryPacksHolder
        = TTypedFeatureValuesHolder<NCB::TBinaryFeaturesPack, EFeatureValuesType::BinaryPack>;
    using TBinaryPacksArrayHolder
        = TCompressedValuesHolderImpl<NCB::TBinaryFeaturesPack, EFeatureValuesType::BinaryPack>;

    template <class T, EFeatureValuesType TType>
    class TPackedBinaryValuesHolderImpl : public TTypedFeatureValuesHolder<T, TType> {
    public:
        using TBase = TTypedFeatureValuesHolder<T, TType>;

    public:
        TPackedBinaryValuesHolderImpl(ui32 featureId, const TBinaryPacksHolder* packsData, ui8 bitIdx)
            : TBase(featureId, packsData->GetSize())
            , PacksData(packsData)
            , BitIdx(bitIdx)
        {
            CB_ENSURE(
                BitIdx < sizeof(NCB::TBinaryFeaturesPack) * CHAR_BIT,
                "BitIdx=" << BitIdx << " is bigger than limit ("
                << sizeof(NCB::TBinaryFeaturesPack) * CHAR_BIT << ')'
            );
        }

        ui8 GetBitIdx() const {
            return BitIdx;
        }

        // in some cases non-standard T can be useful / more efficient
        template <class T2 = T>
        TMaybeOwningArrayHolder<T> ExtractValuesT(NPar::TLocalExecutor* localExecutor) const {
            Y_UNUSED(localExecutor);

            if (const auto* packsArrayData = dynamic_cast<const TBinaryPacksArrayHolder*>(PacksData)) {
                TVector<T> dst;
                dst.yresize(this->GetSize());
                TArrayRef<T> dstRef(dst);

                NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << BitIdx;

                auto visitor = [dstRef, bitIdx = BitIdx, bitMask](ui32 objectIdx, NCB::TBinaryFeaturesPack pack) {
                    dstRef[objectIdx] = (pack & bitMask) >> bitIdx;
                };
                packsArrayData->ForEach(visitor);

                return TMaybeOwningArrayHolder<T>::CreateOwning(std::move(dst));
            } else {
                Y_FAIL("PacksData is not TBinaryPacksArrayHolder");
            }
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            return ExtractValuesT<T>(localExecutor);
        }

    private:
        const TBinaryPacksHolder* PacksData;
        ui8 BitIdx;
    };


    using TExclusiveFeatureBundleHolder
        = TTypedFeatureValuesHolder<ui16, EFeatureValuesType::ExclusiveFeatureBundle>;
    using TExclusiveFeatureBundleArrayHolder
        = TCompressedValuesHolderImpl<ui16, EFeatureValuesType::ExclusiveFeatureBundle>;


    template <class T, EFeatureValuesType TType>
    class TBundlePartValuesHolderImpl : public TTypedFeatureValuesHolder<T, TType> {
    public:
        using TBase = TTypedFeatureValuesHolder<T, TType>;

    public:
        TBundlePartValuesHolderImpl(ui32 featureId,
                                    const TExclusiveFeatureBundleHolder* bundlesData,
                                    NCB::TBoundsInBundle boundsInBundle)
            : TBase(featureId, bundlesData->GetSize())
            , BundlesData(bundlesData)
            , BundleSizeInBytes(0) // inited below
            , BoundsInBundle(boundsInBundle)
        {
            CB_ENSURE_INTERNAL(bundlesData, "bundlesData is empty");

            ui32 bitsPerKey;
            if (const auto* denseData = dynamic_cast<const TExclusiveFeatureBundleArrayHolder*>(BundlesData)) {
                bitsPerKey = denseData->GetBitsPerKey();
            } else {
                CB_ENSURE_INTERNAL(false, "Unsupported bundlesData type");
            }
            CB_ENSURE_INTERNAL(
                (bitsPerKey == CHAR_BIT) || (bitsPerKey == (2 * CHAR_BIT)),
                "Unsupported " << LabeledOutput(bitsPerKey)
            );
            BundleSizeInBytes = bitsPerKey / CHAR_BIT;

            const ui32 maxBound = ui32(1) << bitsPerKey;
            CB_ENSURE_INTERNAL(
                (boundsInBundle.Begin < boundsInBundle.End),
                LabeledOutput(boundsInBundle) << " do not represent a valid range"
            );
            CB_ENSURE_INTERNAL(boundsInBundle.End <= maxBound, "boundsInBundle.End > maxBound");
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            switch (BundleSizeInBytes) {
                case 1:
                    return ExtractValuesImpl<ui8>(localExecutor);
                case 2:
                    return ExtractValuesImpl<ui16>(localExecutor);
                default:
                    Y_UNREACHABLE();
            }
            Y_UNREACHABLE();
        }

        ui32 GetBundleSizeInBytes() const {
            return BundleSizeInBytes;
        }

        NCB::TBoundsInBundle GetBoundsInBundle() const {
            return BoundsInBundle;
        }

    private:
        template <class TBundle>
        TMaybeOwningArrayHolder<T> ExtractValuesImpl(NPar::TLocalExecutor* localExecutor) const {
            if (const auto* bundlesArrayData
                = dynamic_cast<const TExclusiveFeatureBundleArrayHolder*>(BundlesData))
            {
                TVector<T> dst;
                dst.yresize(this->GetSize());
                TArrayRef<T> dstRef(dst);

                auto visitor = [dstRef, boundsInBundle = BoundsInBundle] (ui32 objectIdx, auto bundle) {
                    dstRef[objectIdx] = GetBinFromBundle<T>(bundle, boundsInBundle);
                };
                bundlesArrayData->GetArrayData<TBundle>().ParallelForEach(visitor, localExecutor);

                return TMaybeOwningArrayHolder<T>::CreateOwning(std::move(dst));
            } else {
                Y_FAIL("PacksData is not TBundlesArrayData");
            }
        }

    private:
        const TExclusiveFeatureBundleHolder* BundlesData;
        ui32 BundleSizeInBytes;
        NCB::TBoundsInBundle BoundsInBundle;
    };

    using TFeaturesGroupHolder
        = TTypedFeatureValuesHolder<ui8, EFeatureValuesType::FeaturesGroup>;
    using TFeaturesGroupArrayHolder
        = TCompressedValuesHolderImpl<ui8, EFeatureValuesType::FeaturesGroup>;

    template <class T, EFeatureValuesType TType> // T - is always ui8 now
    class TFeaturesGroupPartValuesHolderImpl : public TTypedFeatureValuesHolder<T, TType> {
    public:
        using TBase = TTypedFeatureValuesHolder<T, TType>;

    public:
        TFeaturesGroupPartValuesHolderImpl(ui32 featureId,
                                           const TFeaturesGroupHolder* groupData,
                                           ui32 inGroupIdx)
            : TBase(featureId, groupData->GetSize())
            , GroupData(groupData)
            , GroupSizeInBytes(0) // inited below
            , InGroupIdx(inGroupIdx)
        {
            CB_ENSURE_INTERNAL(groupData, "groupData is empty");
            ui32 bitsPerKey;
            if (const auto* denseData = dynamic_cast<const TFeaturesGroupArrayHolder*>(GroupData)) {
                bitsPerKey = denseData->GetBitsPerKey();
            } else {
                CB_ENSURE_INTERNAL(false, "Unsupported groupData type");
            }
            CB_ENSURE_INTERNAL(
                (bitsPerKey == CHAR_BIT) || (bitsPerKey == 2 * CHAR_BIT) || (bitsPerKey == 4 * CHAR_BIT),
                "Unsupported " << LabeledOutput(bitsPerKey)
            );
            GroupSizeInBytes = bitsPerKey / CHAR_BIT;
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            switch(GroupSizeInBytes) {
                case 1:
                    return ExtractValuesImpl<ui8>(localExecutor);
                case 2:
                    return ExtractValuesImpl<ui16>(localExecutor);
                case 4:
                    return ExtractValuesImpl<ui32>(localExecutor);
                default:
                    Y_UNREACHABLE();
            }
            Y_UNREACHABLE();
        }

    private:
        template <typename TGroup>
        TMaybeOwningArrayHolder<T> ExtractValuesImpl(NPar::TLocalExecutor* localExecutor) const {
            if (const auto* bundlesArrayData = dynamic_cast<const TFeaturesGroupArrayHolder*>(GroupData)) {
                TVector<T> dst;
                dst.yresize(this->GetSize());
                TArrayRef<T> dstRef(dst);

                auto visitor = [dstRef, firstBitPos = InGroupIdx * CHAR_BIT](ui32 objectIdx, TGroup group) {
                    dstRef[objectIdx] = (group >> firstBitPos);
                };

                bundlesArrayData->GetArrayData<TGroup>().ParallelForEach(visitor, localExecutor);

                return TMaybeOwningArrayHolder<T>::CreateOwning(std::move(dst));
            } else {
                Y_FAIL("GroupsData is not TFeaturesGroupArrayData");
            }
        }

    private:
        const TFeaturesGroupHolder* GroupData;
        ui32 GroupSizeInBytes;
        ui32 InGroupIdx;
    };

    using IQuantizedFloatValuesHolder = TTypedFeatureValuesHolder<ui8, EFeatureValuesType::QuantizedFloat>;

    using ICloneableQuantizedFloatValuesHolder
        = TCloneableWithSubsetIndexingValuesHolder<ui8, EFeatureValuesType::QuantizedFloat>;

    using TQuantizedFloatValuesHolder
        = TCompressedValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;
    using TQuantizedFloatPackedBinaryValuesHolder
        = TPackedBinaryValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;
    using TQuantizedFloatBundlePartValuesHolder
        = TBundlePartValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;
    using TQuantizedFloatGroupPartValuesHolder
        = TFeaturesGroupPartValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;


    using IQuantizedCatValuesHolder
        = TTypedFeatureValuesHolder<ui32, EFeatureValuesType::PerfectHashedCategorical>;

    using ICloneableQuantizedCatValuesHolder
        = TCloneableWithSubsetIndexingValuesHolder<ui32, EFeatureValuesType::PerfectHashedCategorical>;

    using TQuantizedCatValuesHolder
        = TCompressedValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;
    using TQuantizedCatPackedBinaryValuesHolder
        = TPackedBinaryValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;
    using TQuantizedCatBundlePartValuesHolder
        = TBundlePartValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;


    using TTokenizedTextValuesHolder = TTypedFeatureValuesHolder<TText, EFeatureValuesType::TokenizedText>;
    using TTokenizedTextArrayValuesHolder = TArrayValuesHolder<TText, EFeatureValuesType::TokenizedText>;

}
