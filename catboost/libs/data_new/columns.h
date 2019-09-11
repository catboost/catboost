#pragma once

#include "exclusive_feature_bundling.h"
#include "feature_grouping.h"
#include "features_layout.h"
#include "packed_binary_features.h"

#include <catboost/libs/data_types/text.h>
#include <catboost/libs/helpers/array_subset.h>
#include <catboost/libs/helpers/compression.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/helpers/polymorphic_type_containers.h>
#include <catboost/libs/helpers/resource_constrained_executor.h>
#include <catboost/libs/helpers/sparse_array.h>

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
    using TFeaturesSparseArrayIndexing = TSparseArrayIndexing<ui32>;

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


    template <class T, EFeatureValuesType TType>
    struct TValuesHolderWithScheduleGetSubset : public TTypedFeatureValuesHolder<T, TType>
    {
        TValuesHolderWithScheduleGetSubset(ui32 featureId, ui32 size, bool isSparse)
            : TTypedFeatureValuesHolder<T, TType>(featureId, size, isSparse)
        {}

        /* getting subset might require additional data, so use TResourceConstrainedExecutor
         */
        virtual void ScheduleGetSubset(
            // pointer to capture in lambda
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<TTypedFeatureValuesHolder<T, TType>>* subsetDst
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


    template <class T, EFeatureValuesType TType>
    class TSparsePolymorphicArrayValuesHolder: public TValuesHolderWithScheduleGetSubset<T, TType> {
    public:
        TSparsePolymorphicArrayValuesHolder(ui32 featureId, TConstPolymorphicValuesSparseArray<T, ui32>&& data)
            : TValuesHolderWithScheduleGetSubset<T, TType>(featureId, data.GetSize(), /*isSparse*/ true)
            , Data(std::move(data))
        {}

        void ScheduleGetSubset(
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<TTypedFeatureValuesHolder<T, TType>>* subsetDst
        ) const override {
            resourceConstrainedExecutor->Add(
                {
                    Data.EstimateGetSubsetCpuRamUsage(*subsetInvertedIndexing),
                    [this, subsetInvertedIndexing, subsetDst] () {
                        *subsetDst = MakeHolder<TSparsePolymorphicArrayValuesHolder>(
                            this->GetId(),
                            this->GetData().GetSubset(*subsetInvertedIndexing)
                        );
                    }
                }
            );
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            return TMaybeOwningArrayHolder<T>::CreateOwning(Data.ExtractValues());
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            return MakeHolder<typename TConstPolymorphicValuesSparseArray<T, ui32>::TBlockIterator>(
                Data.GetBlockIterator(offset)
            );
        }

        const TConstPolymorphicValuesSparseArray<T, ui32>& GetData() const {
            return Data;
        }

    private:
        TConstPolymorphicValuesSparseArray<T, ui32> Data;
    };


    using TFloatValuesHolder = TTypedFeatureValuesHolder<float, EFeatureValuesType::Float>;
    using TFloatArrayValuesHolder = TPolymorphicArrayValuesHolder<float, EFeatureValuesType::Float>;
    using TFloatSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<float, EFeatureValuesType::Float>;

    using THashedCatValuesHolder = TTypedFeatureValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;
    using THashedCatArrayValuesHolder = TPolymorphicArrayValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;
    using THashedCatSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<ui32, EFeatureValuesType::HashedCategorical>;

    using TStringTextValuesHolder = TTypedFeatureValuesHolder<TString, EFeatureValuesType::StringText>;
    using TStringTextArrayValuesHolder = TPolymorphicArrayValuesHolder<TString, EFeatureValuesType::StringText>;
    using TStringTextSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<TString, EFeatureValuesType::StringText>;


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

    template <class T, EFeatureValuesType TType>
    class TSparseCompressedValuesHolderImpl : public TValuesHolderWithScheduleGetSubset<T, TType> {
    public:
        using TBase = TValuesHolderWithScheduleGetSubset<T, TType>;

    public:
        TSparseCompressedValuesHolderImpl(ui32 featureId, TSparseCompressedArray<T, ui32>&& data)
            : TBase(featureId, data.GetSize(), /*isSparse*/ true)
            , Data(std::move(data))
        {}

        void ScheduleGetSubset(
            const TFeaturesArraySubsetInvertedIndexing* subsetInvertedIndexing,
            TResourceConstrainedExecutor* resourceConstrainedExecutor,
            THolder<TTypedFeatureValuesHolder<T, TType>>* subsetDst
        ) const override {
            resourceConstrainedExecutor->Add(
                {
                    Data.EstimateGetSubsetCpuRamUsage(*subsetInvertedIndexing),
                    [this, subsetInvertedIndexing, subsetDst] () {
                        *subsetDst = MakeHolder<TSparseCompressedValuesHolderImpl<T, TType>>(
                            this->GetId(),
                            this->GetData().GetSubset(*subsetInvertedIndexing)
                        );
                    }
                }
            );
        }

        TMaybeOwningArrayHolder<T> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
            Y_UNUSED(localExecutor);
            return TMaybeOwningArrayHolder<T>::CreateOwning(Data.ExtractValues());
        }

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            return MakeHolder<typename TSparseCompressedArray<T, ui32>::TBlockIterator>(
                Data.GetBlockIterator(offset)
            );
        }

        const TSparseCompressedArray<T, ui32>& GetData() const {
            return Data;
        }

    private:
        TSparseCompressedArray<T, ui32> Data;
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
            : TBase(featureId, packsData->GetSize(), packsData->GetIsSparse())
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

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            if (const auto* packsArrayData = dynamic_cast<const TBinaryPacksArrayHolder*>(PacksData)) {
                auto compressedArrayData = packsArrayData->GetCompressedData();
                const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
                const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

                NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << BitIdx;

                auto transformer = [bitIdx = BitIdx, bitMask](TBinaryFeaturesPack pack) {
                    return (pack & bitMask) >> bitIdx;
                };

                return
                    MakeHolder<TTransformArrayBlockIterator<T, TBinaryFeaturesPack, decltype(transformer)>>(
                        compressedArray.GetRawArray<const TBinaryFeaturesPack>().subspan(
                            subsetIndexing->GetConsecutiveSubsetBeginNonChecked() + offset
                        ),
                        std::move(transformer)
                    );
            } else {
                Y_FAIL("PacksData is not TBinaryPacksArrayHolder");
            }
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
            : TBase(featureId, bundlesData->GetSize(), bundlesData->GetIsSparse())
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

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            if (const auto* bundlesArrayData
                = dynamic_cast<const TExclusiveFeatureBundleArrayHolder*>(BundlesData))
            {
                auto compressedArrayData = bundlesArrayData->GetCompressedData();
                const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
                const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

                IDynamicBlockIteratorPtr<T> result;

                DispatchBitsPerKeyToDataType(
                    compressedArray,
                    "TBundlePartValuesHolderImpl::GetBlockIterator",
                    [&] (const auto* histogram) {
                        using TBundle = std::remove_cvref_t<decltype(*histogram)>;

                        auto transformer = [boundsInBundle = BoundsInBundle] (TBundle bundle) {
                            return GetBinFromBundle<T>(bundle, boundsInBundle);
                        };

                        result = MakeHolder<TTransformArrayBlockIterator<T, TBundle, decltype(transformer)>>(
                            TArrayRef(
                                histogram + subsetIndexing->GetConsecutiveSubsetBeginNonChecked() + offset,
                                TBase::GetSize() - offset
                            ),
                            std::move(transformer)
                        );
                    }
                );

                return result;
            } else {
                Y_FAIL("BundlesData is not TBundlesArrayData");
            }
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
            : TBase(featureId, groupData->GetSize(), groupData->GetIsSparse())
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

        IDynamicBlockIteratorPtr<T> GetBlockIterator(ui32 offset = 0) const override {
            if (const auto* groupsArrayData = dynamic_cast<const TFeaturesGroupArrayHolder*>(GroupData)) {
                auto compressedArrayData = groupsArrayData->GetCompressedData();
                const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
                const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

                IDynamicBlockIteratorPtr<T> result;

                DispatchBitsPerKeyToDataType(
                    compressedArray,
                    "TFeaturesGroupPartValuesHolderImpl::GetBlockIterator",
                    [&] (const auto* histogram) {
                        using TGroup = std::remove_cvref_t<decltype(*histogram)>;

                        auto transformer = [firstBitPos = InGroupIdx * CHAR_BIT] (TGroup group) {
                            return group >> firstBitPos;
                        };

                        result = MakeHolder<TTransformArrayBlockIterator<T, TGroup, decltype(transformer)>>(
                            TArrayRef(
                                histogram + subsetIndexing->GetConsecutiveSubsetBeginNonChecked() + offset,
                                TBase::GetSize() - offset
                            ),
                            std::move(transformer)
                        );
                    }
                );

                return result;
            } else {
                Y_FAIL("GroupsData is not TFeaturesGroupArrayData");
            }
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
    using TQuantizedFloatSparseValuesHolder
        = TSparseCompressedValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;
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
    using TQuantizedCatSparseValuesHolder
        = TSparseCompressedValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;
    using TQuantizedCatPackedBinaryValuesHolder
        = TPackedBinaryValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;
    using TQuantizedCatBundlePartValuesHolder
        = TBundlePartValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;


    using TTokenizedTextValuesHolder = TTypedFeatureValuesHolder<TText, EFeatureValuesType::TokenizedText>;
    using TTokenizedTextArrayValuesHolder = TPolymorphicArrayValuesHolder<TText, EFeatureValuesType::TokenizedText>;
    using TTokenizedTextSparseValuesHolder = TSparsePolymorphicArrayValuesHolder<TText, EFeatureValuesType::TokenizedText>;

}
