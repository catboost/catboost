#pragma once

#include "columns.h"

#include "feature_grouping.h"
#include "exclusive_feature_bundling.h"
#include "packed_binary_features.h"


namespace NCB {

    struct ICompositeValuesHolder : public IFeatureValuesHolder {
        ICompositeValuesHolder(EFeatureValuesType featureValuesType,
                             ui32 featureId,
                             ui32 size)
            : IFeatureValuesHolder(featureValuesType, featureId, size)
        {}

    };

    template <typename T, EFeatureValuesType ValuesType>
    using ICompositeColumnTemplate = IQuantizedFeatureValuesHolder<T, ValuesType, ICompositeValuesHolder>;

    using IBinaryPacksArray = ICompositeColumnTemplate<NCB::TBinaryFeaturesPack, EFeatureValuesType::BinaryPack>;
    using IFeaturesGroupArray = ICompositeColumnTemplate<ui8, EFeatureValuesType::FeaturesGroup>;
    using IExclusiveFeatureBundleArray = ICompositeColumnTemplate<ui16, EFeatureValuesType::ExclusiveFeatureBundle>;

    using TBinaryPacksArrayHolder = TCompressedValuesHolderImpl<IBinaryPacksArray>;
    using TFeaturesGroupArrayHolder = TCompressedValuesHolderImpl<IFeaturesGroupArray>;
    using TExclusiveFeatureBundleArrayHolder = TCompressedValuesHolderImpl<IExclusiveFeatureBundleArray>;


    template <class TBase>
    class TPackedBinaryValuesHolderImpl : public TBase {
    public:
        TPackedBinaryValuesHolderImpl(ui32 featureId, const IBinaryPacksArray* packsData, ui8 bitIdx)
            : TBase(featureId, packsData->GetSize())
            , PacksData(dynamic_cast<const TBinaryPacksArrayHolder*>(packsData))
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

        bool IsSparse() const {
            return PacksData->IsSparse();
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            Y_UNUSED(cloningParams);
            CB_ENSURE_INTERNAL(false, "TPackedBinaryValuesHolderImpl cloning not implemented");
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(cloningParams);
            Y_UNUSED(localExecutor);
            CB_ENSURE_INTERNAL(false, "TPackedBinaryValuesHolderImpl cloning not implemented");
        }

        // in some cases non-standard T can be useful / more efficient
        template <class T2 = typename TBase::TValueType>
        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValuesT(NPar::TLocalExecutor* localExecutor) const {
            Y_UNUSED(localExecutor);
            TVector<typename TBase::TValueType> dst;
            dst.yresize(this->GetSize());
            TArrayRef<typename TBase::TValueType> dstRef(dst);

            NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << BitIdx;

            auto visitor = [dstRef, bitIdx = BitIdx, bitMask](ui32 objectIdx, NCB::TBinaryFeaturesPack pack) {
                dstRef[objectIdx] = (pack & bitMask) >> bitIdx;
            };
            PacksData->ForEach(visitor);

            return TMaybeOwningArrayHolder<typename TBase::TValueType>::CreateOwning(std::move(dst));
        }

        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValues(NPar::TLocalExecutor* localExecutor) const {
            return ExtractValuesT<typename TBase::TValueType>(localExecutor);
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override {
            auto compressedArrayData = PacksData->GetCompressedData();
            const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
            const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

            NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << BitIdx;

            auto transformer = [bitIdx = BitIdx, bitMask](TBinaryFeaturesPack pack) {
                return (pack & bitMask) >> bitIdx;
            };
            return MakeTransformingArraySubsetBlockIterator<typename TBase::TValueType>(
                subsetIndexing,
                compressedArray.GetRawArray<const TBinaryFeaturesPack>(),
                offset,
                std::move(transformer)
            );
        }

    private:
        const TBinaryPacksArrayHolder* PacksData;
        ui8 BitIdx;
    };

    template <class TBase>
    class TBundlePartValuesHolderImpl : public TBase {
    public:
        TBundlePartValuesHolderImpl(ui32 featureId,
                                    const IExclusiveFeatureBundleArray* bundlesData,
                                    NCB::TBoundsInBundle boundsInBundle)
            : TBase(featureId, bundlesData->GetSize())
            , BundlesData(dynamic_cast<const TExclusiveFeatureBundleArrayHolder*>(bundlesData))
            , BundleSizeInBytes(0) // inited below
            , BoundsInBundle(boundsInBundle)
        {
            CB_ENSURE_INTERNAL(bundlesData, "bundlesData is empty");
            CB_ENSURE_INTERNAL(BundlesData, "Expected TExclusiveFeatureBundleArrayHolder");
            ui32 bitsPerKey;

            bitsPerKey = BundlesData->GetBitsPerKey();

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

        bool IsSparse() const override {
            return BundlesData->IsSparse();
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            Y_UNUSED(cloningParams);
            CB_ENSURE_INTERNAL(false, "TBundlePartValuesHolderImpl cloning not implemented");
            return 0;
        }


        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(cloningParams);
            Y_UNUSED(localExecutor);
            CB_ENSURE_INTERNAL(false, "TBundlePartValuesHolderImpl cloning not implemented");
        }

        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
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

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const  override {
            auto compressedArrayData = BundlesData->GetCompressedData();
            const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
            const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

            IDynamicBlockIteratorPtr<typename TBase::TValueType> result;

            compressedArray.DispatchBitsPerKeyToDataType(
                "TBundlePartValuesHolderImpl::GetBlockIterator",
                [&] (const auto* histogram) {
                    using TBundle = std::remove_cvref_t<decltype(*histogram)>;

                    auto transformer = [boundsInBundle = BoundsInBundle] (TBundle bundle) {
                        return GetBinFromBundle<typename TBase::TValueType>(bundle, boundsInBundle);
                    };

                    result = MakeTransformingArraySubsetBlockIterator<typename TBase::TValueType>(
                        subsetIndexing,
                        TArrayRef(
                            histogram,
                            compressedArray.GetSize()
                        ),
                        offset,
                        std::move(transformer)
                    );
                }
            );

            return result;
        }

        ui32 GetBundleSizeInBytes() const {
            return BundleSizeInBytes;
        }

        NCB::TBoundsInBundle GetBoundsInBundle() const {
            return BoundsInBundle;
        }

    private:
        template <class TBundle>
        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValuesImpl(NPar::TLocalExecutor* localExecutor) const {
            TVector<typename TBase::TValueType> dst;
            dst.yresize(this->GetSize());
            TArrayRef<typename TBase::TValueType> dstRef(dst);

            auto visitor = [dstRef, boundsInBundle = BoundsInBundle] (ui32 objectIdx, auto bundle) {
                dstRef[objectIdx] = GetBinFromBundle<typename TBase::TValueType>(bundle, boundsInBundle);
            };
            BundlesData->GetArrayData<TBundle>().ParallelForEach(visitor, localExecutor);
            return TMaybeOwningArrayHolder<typename TBase::TValueType>::CreateOwning(std::move(dst));
        }

    private:
        const TExclusiveFeatureBundleArrayHolder* BundlesData;
        ui32 BundleSizeInBytes;
        NCB::TBoundsInBundle BoundsInBundle;
    };


    template <class TBase> // T - is always ui8 now
    class TFeaturesGroupPartValuesHolderImpl : public TBase {
    public:
        TFeaturesGroupPartValuesHolderImpl(ui32 featureId,
                                           const IFeaturesGroupArray* groupData,
                                           ui32 inGroupIdx)
            : TBase(featureId, groupData->GetSize())
            , GroupData(dynamic_cast<const TFeaturesGroupArrayHolder*>(groupData))
            , GroupSizeInBytes(0) // inited below
            , InGroupIdx(inGroupIdx)
        {
            CB_ENSURE_INTERNAL(GroupData, "groupData is empty or is not TFeaturesGroupArrayHolder");
            ui32 bitsPerKey;
            bitsPerKey = GroupData->GetBitsPerKey();
            CB_ENSURE_INTERNAL(
                (bitsPerKey == CHAR_BIT) || (bitsPerKey == 2 * CHAR_BIT) || (bitsPerKey == 4 * CHAR_BIT),
                "Unsupported " << LabeledOutput(bitsPerKey)
            );
            GroupSizeInBytes = bitsPerKey / CHAR_BIT;
        }

        bool IsSparse() const override {
            return GroupData->IsSparse();
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            Y_UNUSED(cloningParams);
            CB_ENSURE_INTERNAL(false, "TFeaturesGroupPartValuesHolderImpl cloning not implemented");
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(cloningParams);
            Y_UNUSED(localExecutor);
            CB_ENSURE_INTERNAL(false, "TFeaturesGroupPartValuesHolderImpl cloning not implemented");
        }

        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValues(NPar::TLocalExecutor* localExecutor) const override {
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

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override {
            auto compressedArrayData = GroupData->GetCompressedData();
            const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
            const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

            IDynamicBlockIteratorPtr<typename TBase::TValueType> result;

            compressedArray.DispatchBitsPerKeyToDataType(
                "TFeaturesGroupPartValuesHolderImpl::GetBlockIterator",
                [&] (const auto* histogram) {
                    using TGroup = std::remove_cvref_t<decltype(*histogram)>;

                    auto transformer = [firstBitPos = InGroupIdx * CHAR_BIT] (TGroup group) {
                        return group >> firstBitPos;
                    };

                    result = MakeTransformingArraySubsetBlockIterator<typename TBase::TValueType>(
                        subsetIndexing,
                        TArrayRef(
                            histogram,
                            compressedArray.GetSize()
                        ),
                        offset,
                        std::move(transformer)
                    );
                }
            );
            return result;
        }

    private:
        template <typename TGroup>
        TMaybeOwningArrayHolder<typename TBase::TValueType> ExtractValuesImpl(NPar::TLocalExecutor* localExecutor) const {
            TVector<typename TBase::TValueType> dst;
            dst.yresize(this->GetSize());
            TArrayRef<typename TBase::TValueType> dstRef(dst);

            auto visitor = [dstRef, firstBitPos = InGroupIdx * CHAR_BIT](ui32 objectIdx, TGroup group) {
                dstRef[objectIdx] = (group >> firstBitPos);
            };

            GroupData->GetArrayData<TGroup>().ParallelForEach(visitor, localExecutor);

            return TMaybeOwningArrayHolder<typename TBase::TValueType>::CreateOwning(std::move(dst));
        }

    private:
        const TFeaturesGroupArrayHolder* GroupData;
        ui32 GroupSizeInBytes;
        ui32 InGroupIdx;
    };


    using TQuantizedFloatPackedBinaryValuesHolder = TPackedBinaryValuesHolderImpl<IQuantizedFloatValuesHolder>;
    using TQuantizedCatPackedBinaryValuesHolder = TPackedBinaryValuesHolderImpl<IQuantizedCatValuesHolder>;

    using TQuantizedFloatBundlePartValuesHolder = TBundlePartValuesHolderImpl<IQuantizedFloatValuesHolder>;
    using TQuantizedCatBundlePartValuesHolder = TBundlePartValuesHolderImpl<IQuantizedCatValuesHolder>;

    using TQuantizedFloatGroupPartValuesHolder = TFeaturesGroupPartValuesHolderImpl<IQuantizedFloatValuesHolder>;
}
