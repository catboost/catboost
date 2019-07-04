#pragma once

#include <catboost/libs/data_new/objects.h>
#include <catboost/libs/data_new/model_dataset_compatibility.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/cat_feature/cat_feature.h>


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

    inline const float* GetRawFeatureDataBeginPtr(
        const TRawObjectsDataProvider& rawObjectsData,
        ui32 consecutiveSubsetBegin,
        ui32 flatFeatureIdx)
    {
        const auto featuresLayout = rawObjectsData.GetFeaturesLayout();
        const ui32 internalFeatureIdx = featuresLayout->GetInternalFeatureIdx(flatFeatureIdx);
        if (featuresLayout->GetExternalFeatureType(flatFeatureIdx) == EFeatureType::Float) {
            return (*(*(**rawObjectsData.GetFloatFeature(internalFeatureIdx)).GetArrayData().GetSrc()
                )).data() + consecutiveSubsetBegin;
        } else {
            return reinterpret_cast<const float*>((*(*(**rawObjectsData.GetCatFeature(internalFeatureIdx))
                .GetArrayData().GetSrc())).data()) + consecutiveSubsetBegin;
        }
    }

    namespace NDetail {
        class TBaseRawFeatureAccessor {
        public:
            TBaseRawFeatureAccessor(
                const TRawObjectsDataProvider& rawObjectsData,
                const TFullModel&
            )
                : RepackedFeaturesHolder(MakeAtomicShared<TVector<TConstArrayRef<float>>>())
                , RawObjectsData(rawObjectsData)
                , ConsecutiveSubsetBegin(GetConsecutiveSubsetBegin(rawObjectsData))
                , RepackedFeaturesRef(*RepackedFeaturesHolder) {}

            const float* GetFeatureDataBeginPtr(ui32 flatFeatureIdx) {
                return GetRawFeatureDataBeginPtr(
                    RawObjectsData,
                    ConsecutiveSubsetBegin,
                    flatFeatureIdx
                );
            }

            Y_FORCE_INLINE auto GetFloatAccessor() {
                return [this](const TFeaturePosition& position, size_t index) -> float {
                    Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < RepackedFeaturesRef.size());
                    Y_ASSERT(SafeIntegerCast<size_t>(index) <
                             RepackedFeaturesRef[position.FlatIndex].size());
                    return RepackedFeaturesRef[position.FlatIndex][index];
                };
            }

            Y_FORCE_INLINE auto GetCatAccessor() {
                 return [this] (const TFeaturePosition& position, size_t index) -> ui32 {
                    Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < RepackedFeaturesRef.size());
                    Y_ASSERT(SafeIntegerCast<size_t>(index) <
                    RepackedFeaturesRef[position.FlatIndex].size());
                    return ConvertFloatCatFeatureToIntHash(
                        RepackedFeaturesRef[position.FlatIndex][index]);

                };
            };

        private:
            TAtomicSharedPtr<TVector<TConstArrayRef<float>>> RepackedFeaturesHolder;
            const TRawObjectsDataProvider& RawObjectsData;
            const ui32 ConsecutiveSubsetBegin;

        protected:
            TVector<TConstArrayRef<float>>& RepackedFeaturesRef;
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
            }

            const ui8* GetFeatureDataBeginPtr(ui32 flatFeatureIdx) {
                BundledIndexesRef[flatFeatureIdx] = QuantizedObjectsData.GetFloatFeatureToExclusiveBundleIndex(
                    TFloatFeatureIdx(flatFeatureIdx));
                PackedIndexesRef[flatFeatureIdx] = QuantizedObjectsData.GetFloatFeatureToPackedBinaryIndex(
                    TFloatFeatureIdx(flatFeatureIdx));

                if (BundledIndexesRef[flatFeatureIdx].Defined()) {
                    return QuantizedObjectsData.GetExclusiveFeaturesBundle(
                        BundledIndexesRef[flatFeatureIdx]->BundleIdx).SrcData.data();
                } else if (PackedIndexesRef[flatFeatureIdx].Defined()) {
                    return (**QuantizedObjectsData.GetBinaryFeaturesPack(
                        PackedIndexesRef[flatFeatureIdx]->PackIdx).GetSrc()).data();
                } else {
                    const auto& featuresLayout = *QuantizedObjectsData.GetFeaturesLayout();
                    CB_ENSURE_INTERNAL(
                        featuresLayout.GetExternalFeatureType(flatFeatureIdx) == EFeatureType::Float,
                        "Mismatched feature type");
                    return (QuantizedObjectsData.GetFloatFeatureRawSrcData(flatFeatureIdx) +
                        ConsecutiveSubsetBegin);
                }
            }
            Y_FORCE_INLINE auto GetFloatAccessor() {
                return [this] (const TFeaturePosition& position, size_t index) -> ui8 {
                    const auto& bundleIdx = BundledIndexesRef[position.Index];
                    const auto& packIdx = PackedIndexesRef[position.Index];
                    ui8 unremappedFeatureBin;

                    if (bundleIdx.Defined()) {
                        const auto& bundleMetaData = BundlesMetaData[bundleIdx->BundleIdx];
                        const auto& bundlePart = bundleMetaData.Parts[bundleIdx->InBundleIdx];
                        auto boundsInBundle = bundlePart.Bounds;

                        const ui8* rawBundlesData = RepackedFeaturesRef[position.FlatIndex].data();

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
                            (RepackedFeaturesRef[position.FlatIndex][index] >> bitIdx) & 1;
                    } else {
                        unremappedFeatureBin = RepackedFeaturesRef[position.FlatIndex][index];
                    }

                    return FloatBinsRemapRef[position.FlatIndex][unremappedFeatureBin];
                };
            }
            Y_FORCE_INLINE auto GetCatAccessor() {
                return [] (const TFeaturePosition& , size_t ) -> ui32 {
                    Y_FAIL();
                    return 0;
                };
            }

        private:
            struct TQuantizedFeaturesAccessorData {
                TVector<TVector<ui8>> FloatBinsRemap;
                TVector<TConstArrayRef<ui8>> RepackedFeatures;
                TVector<TMaybe<TPackedBinaryIndex>> PackedIndexes;
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

        protected:
            TVector<TConstArrayRef<ui8>>& RepackedFeaturesRef;
        };

        template <class TBaseAccesorType>
        class TFeatureAccessorTemplate : public TBaseAccesorType {
            using TBaseAccesorType::RepackedFeaturesRef;
            using TBaseAccesorType::GetFeatureDataBeginPtr;
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
                const auto &featuresLayout = *objectsData.GetFeaturesLayout();
                RepackedFeaturesRef.resize(model.ObliviousTrees->GetFlatFeatureVectorExpectedSize());
                const int objectCount = objectsEnd - objectsBegin;
                if (columnReorderMap.empty()) {
                    for (size_t i = 0; i < model.ObliviousTrees->GetFlatFeatureVectorExpectedSize(); ++i) {
                        if (featuresLayout.GetExternalFeaturesMetaInfo()[i].IsAvailable) {
                            RepackedFeaturesRef[i] =
                                MakeArrayRef(GetFeatureDataBeginPtr(i) + objectsBegin, objectCount);
                        }
                    }
                } else {
                    for (const auto&[origIdx, sourceIdx] : columnReorderMap) {
                        if (featuresLayout.GetExternalFeaturesMetaInfo()[sourceIdx].IsAvailable) {
                            RepackedFeaturesRef[origIdx] =
                                MakeArrayRef(GetFeatureDataBeginPtr(sourceIdx) + objectsBegin, objectCount);
                        }
                    }
                }
            }
        };
    }

    using TRawFeatureAccessor = NDetail::TFeatureAccessorTemplate<NDetail::TBaseRawFeatureAccessor>;
    using TQuantizedFeatureAccessor = NDetail::TFeatureAccessorTemplate<
        NDetail::TBaseQuantizedFeatureAccessor>;
}
