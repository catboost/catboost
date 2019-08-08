#pragma once

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/data_new/model_dataset_compatibility.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/index_range/index_range.h>
#include <catboost/libs/cat_feature/cat_feature.h>

#include <util/generic/cast.h>
#include <util/stream/labeled.h>
#include <util/system/compiler.h>

#include <climits>


namespace NCB {
    template <class TDataProvidersTemplate>
    inline ui32 GetConsecutiveSubsetBegin(const TDataProvidersTemplate& objectsData) {
        const auto maybeConsecutiveSubsetBegin =
            objectsData.GetFeaturesArraySubsetIndexing().GetConsecutiveSubsetBegin();
        CB_ENSURE_INTERNAL(
            maybeConsecutiveSubsetBegin,
            "Only consecutive feature data is supported for apply"
        );
        return *maybeConsecutiveSubsetBegin;
    }

    template <class T, EFeatureValuesType FeatureValuesType>
    inline const float* GetRawFeatureDataBeginPtr(
        TMaybeData<const TTypedFeatureValuesHolder<T, FeatureValuesType>*> column,
        ui32 consecutiveSubsetBegin)
    {
        const TMaybeOwningArrayHolder<const T> fullData =
            *((dynamic_cast<const TArrayValuesHolder<T, FeatureValuesType>&>(**column)).GetArrayData()
                .GetSrc()
        );
        return reinterpret_cast<const float*>((*fullData).data()) + consecutiveSubsetBegin;
    }

    // raw data array is returned, in case of compressed data using external bitsPerKey might be necessary
    template <class T, EFeatureValuesType FeatureValuesType>
    TMaybeOwningArrayHolder<ui8> GetConsecutiveSubRangeColumnData(
        const TTypedFeatureValuesHolder<T, FeatureValuesType>& column,
        ui32 consecutiveSubsetBegin,
        TIndexRange<ui32> indexRange)
    {
        if (const auto* arrayValuesHolder
                = dynamic_cast<const TArrayValuesHolder<T, FeatureValuesType>*>(&column))
        {
            const auto& arrayData = arrayValuesHolder->GetArrayData();
            const auto& fullDataHolder = *arrayData.GetSrc();

            return TMaybeOwningArrayHolder<ui8>::CreateNonOwning(
                MakeArrayRef(
                    (ui8*)(fullDataHolder.data() + consecutiveSubsetBegin + indexRange.Begin),
                    indexRange.GetSize() * sizeof(T)
                )
            );
        } else if (const auto* compressedArrayValuesHolder
                       = dynamic_cast<const TCompressedValuesHolderImpl<T, FeatureValuesType>*>(&column))
        {
            const auto& compressedArrayData = *(compressedArrayValuesHolder->GetCompressedData().GetSrc());

            const ui32 bitsPerKey = compressedArrayData.GetBitsPerKey();

            CB_ENSURE_INTERNAL(!(bitsPerKey % CHAR_BIT), "unsupported " << LabeledOutput(bitsPerKey));

            const ui32 bytesPerKey = bitsPerKey / CHAR_BIT;

            return TMaybeOwningArrayHolder<ui8>::CreateNonOwning(
                MakeArrayRef(
                    (ui8*)(
                        compressedArrayData.GetRawPtr()
                        + (consecutiveSubsetBegin + indexRange.Begin) * bytesPerKey),
                    indexRange.GetSize() * bytesPerKey
                )
            );
        } else {
            CB_ENSURE_INTERNAL(false, "GetConsecutiveSubRangeColumnData: unsupported column type");
        }
        Y_UNREACHABLE();
    }


    inline const float* GetRawFeatureDataBeginPtr(
        const TRawObjectsDataProvider& rawObjectsData,
        ui32 consecutiveSubsetBegin,
        ui32 flatFeatureIdx)
    {
        const auto featuresLayout = rawObjectsData.GetFeaturesLayout();
        const ui32 internalFeatureIdx = featuresLayout->GetInternalFeatureIdx(flatFeatureIdx);
        if (featuresLayout->GetExternalFeatureType(flatFeatureIdx) == EFeatureType::Float) {
            return GetRawFeatureDataBeginPtr(
                rawObjectsData.GetFloatFeature(internalFeatureIdx),
                consecutiveSubsetBegin
            );
        } else {
            return GetRawFeatureDataBeginPtr(
                rawObjectsData.GetCatFeature(internalFeatureIdx),
                consecutiveSubsetBegin
            );
        }
    }

    namespace NDetail {
        class TBaseRawFeatureAccessor {
        public:
            TBaseRawFeatureAccessor(
                const TRawObjectsDataProvider& rawObjectsData,
                const TFullModel& model
            )
                : RepackedFeaturesHolder(MakeAtomicShared<TVector<TMaybeOwningArrayHolder<float>>>())
                , RawObjectsData(rawObjectsData)
                , ConsecutiveSubsetBegin(GetConsecutiveSubsetBegin(rawObjectsData))
                , RepackedFeaturesRef(*RepackedFeaturesHolder)
            {
                RepackedFeaturesRef.resize(model.ObliviousTrees->GetFlatFeatureVectorExpectedSize());
            }

            void AddFeature(
                ui32 sourceFlatFeatureIdx,
                ui32 repackedFlatFeatureIdx,
                TIndexRange<ui32> objectRange)
            {
                const auto featuresLayout = RawObjectsData.GetFeaturesLayout();
                const ui32 internalFeatureIdx = featuresLayout->GetInternalFeatureIdx(sourceFlatFeatureIdx);

                TMaybeOwningArrayHolder<ui8> rawData;

                auto getRawData = [&] (const auto& column) {
                    return GetConsecutiveSubRangeColumnData(column, ConsecutiveSubsetBegin, objectRange);
                };

                if (featuresLayout->GetExternalFeatureType(sourceFlatFeatureIdx) == EFeatureType::Float) {
                    rawData = getRawData(**RawObjectsData.GetFloatFeature(internalFeatureIdx));
                } else {
                    rawData = getRawData(**RawObjectsData.GetCatFeature(internalFeatureIdx));
                }
                if (repackedFlatFeatureIdx >= RepackedFeaturesRef.size()) {
                    RepackedFeaturesRef.resize(repackedFlatFeatureIdx + 1);
                }
                RepackedFeaturesRef[repackedFlatFeatureIdx]
                    = TMaybeOwningArrayHolder<float>::CreateOwningReinterpretCast(rawData);
            }

            Y_FORCE_INLINE auto GetFloatAccessor() {
                return [this](TFeaturePosition position, size_t index) -> float {
                    Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < RepackedFeaturesRef.size());
                    Y_ASSERT(SafeIntegerCast<size_t>(index) <
                             RepackedFeaturesRef[position.FlatIndex].GetSize());
                    return RepackedFeaturesRef[position.FlatIndex][index];
                };
            }

            Y_FORCE_INLINE auto GetCatAccessor() {
                 return [this] (TFeaturePosition position, size_t index) -> ui32 {
                    Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < RepackedFeaturesRef.size());
                    Y_ASSERT(SafeIntegerCast<size_t>(index) <
                    RepackedFeaturesRef[position.FlatIndex].GetSize());
                    return ConvertFloatCatFeatureToIntHash(
                        RepackedFeaturesRef[position.FlatIndex][index]);

                };
            };

        private:
            TAtomicSharedPtr<TVector<TMaybeOwningArrayHolder<float>>> RepackedFeaturesHolder;
            const TRawObjectsDataProvider& RawObjectsData;
            const ui32 ConsecutiveSubsetBegin;

            TVector<TMaybeOwningArrayHolder<float>>& RepackedFeaturesRef;
        };

        class TBaseQuantizedFeatureAccessor {
        public:
            TBaseQuantizedFeatureAccessor(
                const TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
                const TFullModel& model
            )
                : HeavyDataHolder(MakeAtomicShared<TQuantizedFeaturesAccessorData>())
                , BundlesMetaData(quantizedObjectsData.GetExclusiveFeatureBundlesMetaData())
                , FloatBinsRemapRef(HeavyDataHolder->FloatBinsRemap)
                , PackedIndexesRef(HeavyDataHolder->PackedIndexes)
                , BundledIndexesRef(HeavyDataHolder->BundledIndexes)
                , QuantizedObjectsData(quantizedObjectsData)
                , ConsecutiveSubsetBegin(GetConsecutiveSubsetBegin(quantizedObjectsData))
                , RepackedFeaturesRef(HeavyDataHolder->RepackedFeatures)
            {
                FloatBinsRemapRef = GetFloatFeaturesBordersRemap(
                    model, *quantizedObjectsData.GetQuantizedFeaturesInfo().Get());
                PackedIndexesRef.resize(model.ObliviousTrees->GetFlatFeatureVectorExpectedSize());
                BundledIndexesRef.resize(model.ObliviousTrees->GetFlatFeatureVectorExpectedSize());
                RepackedFeaturesRef.resize(model.ObliviousTrees->GetFlatFeatureVectorExpectedSize());

                HeavyDataHolder->RepackedBundleData.resize(
                    quantizedObjectsData.GetExclusiveFeatureBundlesSize());
                HeavyDataHolder->RepackedBinaryPacksData.resize(
                    quantizedObjectsData.GetBinaryFeaturesPacksSize());
            }

            void AddFeature(
                ui32 sourceFlatFeatureIdx,
                ui32 repackedFlatFeatureIdx,
                TIndexRange<ui32> objectRange)
            {
                BundledIndexesRef[repackedFlatFeatureIdx]
                    = QuantizedObjectsData.GetFloatFeatureToExclusiveBundleIndex(
                        TFloatFeatureIdx(sourceFlatFeatureIdx));
                PackedIndexesRef[repackedFlatFeatureIdx]
                    = QuantizedObjectsData.GetFloatFeatureToPackedBinaryIndex(
                        TFloatFeatureIdx(sourceFlatFeatureIdx));

                auto getColumnSubRangeData = [&] (const auto& column) {
                    return GetConsecutiveSubRangeColumnData(column, ConsecutiveSubsetBegin, objectRange);
                };

                if (BundledIndexesRef[repackedFlatFeatureIdx].Defined()) {
                    auto bundleIdx = BundledIndexesRef[repackedFlatFeatureIdx]->BundleIdx;
                    if (!HeavyDataHolder->RepackedBundleData[bundleIdx].GetSize()) {
                        HeavyDataHolder->RepackedBundleData[bundleIdx] = getColumnSubRangeData(
                            QuantizedObjectsData.GetExclusiveFeaturesBundle(bundleIdx));
                    }
                } else if (PackedIndexesRef[repackedFlatFeatureIdx].Defined()) {
                    auto packIdx = PackedIndexesRef[repackedFlatFeatureIdx]->PackIdx;
                    if (!HeavyDataHolder->RepackedBinaryPacksData[packIdx].GetSize()) {
                        HeavyDataHolder->RepackedBinaryPacksData[packIdx] = getColumnSubRangeData(
                            QuantizedObjectsData.GetBinaryFeaturesPack(packIdx));
                    }
                } else {
                    const auto& featuresLayout = *QuantizedObjectsData.GetFeaturesLayout();
                    CB_ENSURE_INTERNAL(
                        featuresLayout.GetExternalFeatureType(sourceFlatFeatureIdx) == EFeatureType::Float,
                        "Mismatched feature type");
                    HeavyDataHolder->RepackedFeatures[repackedFlatFeatureIdx] = getColumnSubRangeData(
                        **QuantizedObjectsData.GetNonPackedFloatFeature(sourceFlatFeatureIdx));
                }
            }
            Y_FORCE_INLINE auto GetFloatAccessor() {
                return [this] (TFeaturePosition position, size_t index) -> ui8 {
                    const auto& bundleIdx = BundledIndexesRef[position.Index];
                    const auto& packIdx = PackedIndexesRef[position.Index];
                    ui8 unremappedFeatureBin;

                    if (bundleIdx.Defined()) {
                        const auto& bundleMetaData = BundlesMetaData[bundleIdx->BundleIdx];
                        const auto& bundlePart = bundleMetaData.Parts[bundleIdx->InBundleIdx];
                        auto boundsInBundle = bundlePart.Bounds;

                        const ui8* rawBundlesData
                            = (*(HeavyDataHolder->RepackedBundleData[bundleIdx->BundleIdx])).data();

                        switch (bundleMetaData.SizeInBytes) {
                            case 1:
                                unremappedFeatureBin = GetBinFromBundle<ui8>(rawBundlesData[index],
                                                                             boundsInBundle);
                                break;
                            case 2:
                                unremappedFeatureBin = GetBinFromBundle<ui8>(
                                    ((const ui16*)rawBundlesData)[index], boundsInBundle);
                                break;
                            default:
                                CB_ENSURE_INTERNAL(
                                    false,
                                    "unsupported Bundle SizeInBytes = " << bundleMetaData.SizeInBytes);
                        }
                    } else if (packIdx.Defined()) {
                        TBinaryFeaturesPack bitIdx = packIdx->BitIdx;
                        unremappedFeatureBin =
                            (HeavyDataHolder->RepackedBinaryPacksData[packIdx->PackIdx][index] >> bitIdx) & 1;
                    } else {
                        unremappedFeatureBin = RepackedFeaturesRef[position.FlatIndex][index];
                    }

                    return FloatBinsRemapRef[position.FlatIndex][unremappedFeatureBin];
                };
            }
            Y_FORCE_INLINE auto GetCatAccessor() {
                return [] (TFeaturePosition , size_t ) -> ui32 {
                    Y_FAIL();
                    return 0;
                };
            }

        private:
            struct TQuantizedFeaturesAccessorData {
                TVector<TVector<ui8>> FloatBinsRemap;
                TVector<TMaybeOwningArrayHolder<ui8>> RepackedFeatures;

                TVector<TMaybeOwningArrayHolder<ui8>> RepackedBinaryPacksData;
                TVector<TMaybe<TPackedBinaryIndex>> PackedIndexes;

                TVector<TMaybeOwningArrayHolder<ui8>> RepackedBundleData;
                TVector<TMaybe<TExclusiveBundleIndex>> BundledIndexes;
            };

        private:
            TAtomicSharedPtr<TQuantizedFeaturesAccessorData> HeavyDataHolder;
            TConstArrayRef<TExclusiveFeaturesBundle> BundlesMetaData;
            TVector<TVector<ui8>>& FloatBinsRemapRef;
            TVector<TMaybe<TPackedBinaryIndex>>& PackedIndexesRef;
            TVector<TMaybe<TExclusiveBundleIndex>>& BundledIndexesRef;
            const TQuantizedForCPUObjectsDataProvider& QuantizedObjectsData;
            const ui32 ConsecutiveSubsetBegin;

            TVector<TMaybeOwningArrayHolder<ui8>>& RepackedFeaturesRef;
        };

        template <class TBaseAccesorType>
        class TFeatureAccessorTemplate : public TBaseAccesorType {
        public:
            template <class TObjectsDataProviderType>
            TFeatureAccessorTemplate(
                const TFullModel& model,
                const TObjectsDataProviderType& objectsData,
                const THashMap<ui32, ui32>& columnReorderMap,
                int objectsBegin,
                int objectsEnd
            )
                : TBaseAccesorType(objectsData, model)
            {
                TIndexRange<ui32> objectsRange(
                    SafeIntegerCast<ui32>(objectsBegin),
                    SafeIntegerCast<ui32>(objectsEnd));

                const auto &featuresLayout = *objectsData.GetFeaturesLayout();

                auto addFeatureIfAvailable = [&] (auto origIdx, auto sourceIdx) {
                    if (featuresLayout.GetExternalFeaturesMetaInfo()[sourceIdx].IsAvailable) {
                        TBaseAccesorType::AddFeature(sourceIdx, origIdx, objectsRange);
                    }
                };

                if (columnReorderMap.empty()) {
                    for (size_t i = 0; i < model.ObliviousTrees->GetFlatFeatureVectorExpectedSize(); ++i) {
                        addFeatureIfAvailable(i, i);
                    }
                } else {
                    for (const auto&[origIdx, sourceIdx] : columnReorderMap) {
                        addFeatureIfAvailable(origIdx, sourceIdx);
                    }
                }
            }
        };
    }

    using TRawFeatureAccessor = NDetail::TFeatureAccessorTemplate<NDetail::TBaseRawFeatureAccessor>;
    using TQuantizedFeatureAccessor = NDetail::TFeatureAccessorTemplate<
        NDetail::TBaseQuantizedFeatureAccessor>;
}
