#pragma once

#include "columns.h"
#include "exclusive_feature_bundling.h"
#include "feature_grouping.h"
#include "packed_binary_features.h"

namespace NCB {
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

    using TQuantizedFloatPackedBinaryValuesHolder
        = TPackedBinaryValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;
    using TQuantizedFloatBundlePartValuesHolder
        = TBundlePartValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;
    using TQuantizedFloatGroupPartValuesHolder
        = TFeaturesGroupPartValuesHolderImpl<ui8, EFeatureValuesType::QuantizedFloat>;

    using TQuantizedCatValuesHolder
        = TCompressedValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;
    using TQuantizedCatPackedBinaryValuesHolder
        = TPackedBinaryValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;
    using TQuantizedCatBundlePartValuesHolder
        = TBundlePartValuesHolderImpl<ui32, EFeatureValuesType::PerfectHashedCategorical>;
}
