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
        TPackedBinaryValuesHolderImpl(ui32 featureId, THolder<TBinaryPacksArrayHolder>&& packsData, ui8 bitIdx)
            : TPackedBinaryValuesHolderImpl(featureId, packsData.Get(), bitIdx)
        {
            PacksDataHolder = std::move(packsData);
        }

        ui8 GetBitIdx() const {
            return BitIdx;
        }

        bool IsSparse() const override {
            return PacksData->IsSparse();
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            return PacksData->EstimateMemoryForCloning(cloningParams);
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            return MakeHolder<TPackedBinaryValuesHolderImpl>(
                this->GetId(),
                DynamicHolderCast<TBinaryPacksArrayHolder>(
                    PacksData->CloneWithNewSubsetIndexing(cloningParams, localExecutor),
                    "Column type changed after cloning"
                ),
                BitIdx
            );
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override {
            auto compressedArrayData = PacksData->GetCompressedData();
            const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
            const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

            NCB::TBinaryFeaturesPack bitMask = NCB::TBinaryFeaturesPack(1) << BitIdx;

            auto transformer = [bitIdx = BitIdx, bitMask](TBinaryFeaturesPack pack) {
                return (pack & bitMask) >> bitIdx;
            };
            return MakeTransformingArraySubsetBlockIterator<ui8>(
                subsetIndexing,
                compressedArray.GetRawArray<const TBinaryFeaturesPack>(),
                offset,
                std::move(transformer)
            );
        }

    private:
        const TBinaryPacksArrayHolder* PacksData;
        ui8 BitIdx;
        THolder<TBinaryPacksArrayHolder> PacksDataHolder;
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

        TBundlePartValuesHolderImpl(ui32 featureId,
                                    THolder<TExclusiveFeatureBundleArrayHolder>&& bundlesData,
                                    NCB::TBoundsInBundle boundsInBundle)
            : TBundlePartValuesHolderImpl(featureId, bundlesData.Get(), boundsInBundle)
        {
            BundlesDataHolder = std::move(bundlesData);
        }

        bool IsSparse() const override {
            return BundlesData->IsSparse();
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            return BundlesData->EstimateMemoryForCloning(cloningParams);
        }


        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            return MakeHolder<TBundlePartValuesHolderImpl>(
                this->GetId(),
                DynamicHolderCast<TExclusiveFeatureBundleArrayHolder>(
                    BundlesData->CloneWithNewSubsetIndexing(cloningParams, localExecutor),
                    "Column type changed after cloning"
                ),
                BoundsInBundle
            );
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const  override {
            auto compressedArrayData = BundlesData->GetCompressedData();
            const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
            const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

            IDynamicBlockIteratorPtr<ui8> result;

            compressedArray.DispatchBitsPerKeyToDataType(
                "TBundlePartValuesHolderImpl::GetBlockIterator",
                [&] (const auto* histogram) {
                    using TBundle = std::remove_cvref_t<decltype(*histogram)>;

                    auto transformer = [boundsInBundle = BoundsInBundle] (TBundle bundle) {
                        return GetBinFromBundle<ui8>(bundle, boundsInBundle);
                    };

                    result = MakeTransformingArraySubsetBlockIterator<ui8>(
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
        const TExclusiveFeatureBundleArrayHolder* BundlesData;
        ui32 BundleSizeInBytes;
        NCB::TBoundsInBundle BoundsInBundle;
        THolder<TExclusiveFeatureBundleArrayHolder> BundlesDataHolder;
    };


    template <class TBase>
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

        TFeaturesGroupPartValuesHolderImpl(ui32 featureId,
                                           THolder<TFeaturesGroupArrayHolder>&& groupData,
                                           ui32 inGroupIdx)
            : TFeaturesGroupPartValuesHolderImpl(featureId, groupData.Get(), inGroupIdx)
        {
            GroupDataHolder = std::move(groupData);
        }

        bool IsSparse() const override {
            return GroupData->IsSparse();
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            return GroupData->EstimateMemoryForCloning(cloningParams);
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::ILocalExecutor* localExecutor
        ) const override {
            return MakeHolder<TFeaturesGroupPartValuesHolderImpl>(
                this->GetId(),
                DynamicHolderCast<TFeaturesGroupArrayHolder>(
                    GroupData->CloneWithNewSubsetIndexing(cloningParams, localExecutor),
                    "Column type changed after cloning"
                ),
                InGroupIdx
            );
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset = 0) const override {
            auto compressedArrayData = GroupData->GetCompressedData();
            const TCompressedArray& compressedArray = *compressedArrayData.GetSrc();
            const TFeaturesArraySubsetIndexing* subsetIndexing = compressedArrayData.GetSubsetIndexing();

            IDynamicBlockIteratorPtr<ui8> result;

            compressedArray.DispatchBitsPerKeyToDataType(
                "TFeaturesGroupPartValuesHolderImpl::GetBlockIterator",
                [&] (const auto* histogram) {
                    using TGroup = std::remove_cvref_t<decltype(*histogram)>;

                    auto transformer = [firstBitPos = InGroupIdx * CHAR_BIT] (TGroup group) {
                        return group >> firstBitPos;
                    };

                    result = MakeTransformingArraySubsetBlockIterator<ui8>(
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
        const TFeaturesGroupArrayHolder* GroupData;
        ui32 GroupSizeInBytes;
        ui32 InGroupIdx;
        THolder<TFeaturesGroupArrayHolder> GroupDataHolder;
    };


    using TQuantizedFloatPackedBinaryValuesHolder = TPackedBinaryValuesHolderImpl<IQuantizedFloatValuesHolder>;
    using TQuantizedCatPackedBinaryValuesHolder = TPackedBinaryValuesHolderImpl<IQuantizedCatValuesHolder>;

    using TQuantizedFloatBundlePartValuesHolder = TBundlePartValuesHolderImpl<IQuantizedFloatValuesHolder>;
    using TQuantizedCatBundlePartValuesHolder = TBundlePartValuesHolderImpl<IQuantizedCatValuesHolder>;

    using TQuantizedFloatGroupPartValuesHolder = TFeaturesGroupPartValuesHolderImpl<IQuantizedFloatValuesHolder>;
}
