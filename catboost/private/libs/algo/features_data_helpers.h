#pragma once

#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/dynamic_iterator.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/model/model.h>

#include <util/generic/array_ref.h>
#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/system/compiler.h>
#include <util/system/types.h>
#include <util/system/yassert.h>


namespace NCB {

    class TRawFeaturesBlockIterator;
    class TQuantizedFeaturesBlockIterator;

    class IFeaturesBlockIteratorVisitor {
    public:
        virtual ~IFeaturesBlockIteratorVisitor() = default;

        virtual void Visit(const TRawFeaturesBlockIterator& rawFeaturesBlockIterator) {
            Y_UNUSED(rawFeaturesBlockIterator);
        }

        virtual void Visit(const TQuantizedFeaturesBlockIterator& quantizedFeaturesBlockIterator) {
            Y_UNUSED(quantizedFeaturesBlockIterator);
        }
    };

    class IFeaturesBlockIterator {
    public:
        virtual ~IFeaturesBlockIterator() = default;

        virtual void NextBlock(size_t size) = 0;

        virtual void Accept(IFeaturesBlockIteratorVisitor* visitor) const = 0;
    };

    namespace NDetail {

        template <class TObjectsDataProviderType, class TFloatValue, class TCatValue,
                  class TTextValue, class TEmbeddingValue, class TAccessor>
        class TFeaturesBlockIteratorBase : public IFeaturesBlockIterator {
        public:
            TFeaturesBlockIteratorBase(
                const TFullModel& model,
                const TObjectsDataProviderType& objectsData,
                const THashMap<ui32, ui32>& columnReorderMap,
                ui32 objectOffset
            )
                : ObjectsData(objectsData)
            {
                size_t flatFeatureCount = model.ModelTrees->GetFlatFeatureVectorExpectedSize();
                FloatBlockIterators.resize(flatFeatureCount);
                CatBlockIterators.resize(flatFeatureCount);
                TextBlockIterators.resize(flatFeatureCount);
                EmbeddingBlockIterators.resize(flatFeatureCount);
                FloatValues.resize(flatFeatureCount);
                CatValues.resize(flatFeatureCount);
                TextValues.resize(flatFeatureCount);
                EmbeddingValues.resize(flatFeatureCount);

                for (const auto&[modelFlatFeatureIdx, dataFlatFeatureIdx] : columnReorderMap) {
                    AddFeature(modelFlatFeatureIdx, dataFlatFeatureIdx, objectOffset);
                }
            }

            void NextBlock(size_t size) override {
                for (auto flatFeatureIdx : xrange(FloatBlockIterators.size())) {
                    if (FloatBlockIterators[flatFeatureIdx]) {
                        Y_ASSERT(!CatBlockIterators[flatFeatureIdx]);
                        FloatValues[flatFeatureIdx] = FloatBlockIterators[flatFeatureIdx]->Next(size);
                    } else if (CatBlockIterators[flatFeatureIdx]) {
                        CatValues[flatFeatureIdx] = CatBlockIterators[flatFeatureIdx]->Next(size);
                    } else if (TextBlockIterators[flatFeatureIdx]) {
                        TextValues[flatFeatureIdx] = TextBlockIterators[flatFeatureIdx]->Next(size);
                    }  else if (EmbeddingBlockIterators[flatFeatureIdx]) {
                        auto block = EmbeddingBlockIterators[flatFeatureIdx]->Next(size);
                        EmbeddingValues[flatFeatureIdx].resize(block.size());
                        for (auto i : xrange(block.size())) {
                            EmbeddingValues[flatFeatureIdx][i] = *block[i];
                        }
                    }
                }
            }

            void AddFeature(ui32 modelFlatFeatureIdx, ui32 dataFlatFeatureIdx, ui32 objectOffset) {
                const auto featuresLayout = ObjectsData.GetFeaturesLayout();
                const TFeatureMetaInfo& featureMetaInfo
                    = featuresLayout->GetExternalFeaturesMetaInfo()[dataFlatFeatureIdx];

                CB_ENSURE(
                    featureMetaInfo.IsAvailable,
                    "Required feature #" << dataFlatFeatureIdx << " is not available in dataset"
                );

                const ui32 internalFeatureIdx = featuresLayout->GetInternalFeatureIdx(dataFlatFeatureIdx);

                if (featureMetaInfo.Type == EFeatureType::Float) {
                    auto maybeFloatIterator = (*ObjectsData.GetFloatFeature(internalFeatureIdx))
                        ->GetBlockIterator(objectOffset);
                    auto* floatIteratorPtr = dynamic_cast<IDynamicBlockIterator<TFloatValue>*>(maybeFloatIterator.Get());
                    CB_ENSURE_INTERNAL(floatIteratorPtr, "Should be IDynamicBlockIteratorPtr<TFloatValue>");
                    Y_UNUSED(maybeFloatIterator.Release());
                    FloatBlockIterators[modelFlatFeatureIdx] = IDynamicBlockIteratorPtr<TFloatValue>(floatIteratorPtr);
                } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                    auto maybeCatIterator = (*ObjectsData.GetCatFeature(internalFeatureIdx))
                        ->GetBlockIterator(objectOffset);
                    auto* catIteratorPtr = dynamic_cast<IDynamicBlockIterator<TCatValue>*>(maybeCatIterator.Get());
                    CB_ENSURE_INTERNAL(catIteratorPtr, "Should be IDynamicBlockIteratorPtr<TCatValue>");
                    Y_UNUSED(maybeCatIterator.Release());
                    CatBlockIterators[modelFlatFeatureIdx] = IDynamicBlockIteratorPtr<TCatValue>(catIteratorPtr);
                } else if (featureMetaInfo.Type == EFeatureType::Text) {
                    TextBlockIterators[modelFlatFeatureIdx]
                        = (*ObjectsData.GetTextFeature(internalFeatureIdx))
                            ->GetBlockIterator(objectOffset);
                } else if (featureMetaInfo.Type == EFeatureType::Embedding) {
                    EmbeddingBlockIterators[modelFlatFeatureIdx]
                        = (*ObjectsData.GetEmbeddingFeature(internalFeatureIdx))
                            ->GetBlockIterator(objectOffset);
                } else {
                    CB_ENSURE(
                        false,
                        "Applier cannot use feature #" << dataFlatFeatureIdx << " with type "
                            << featureMetaInfo.Type
                    );
                }
            }

            TConstArrayRef<TConstArrayRef<TFloatValue>> GetFloatValues() const {
                return FloatValues;
            }

            TConstArrayRef<TConstArrayRef<TCatValue>> GetCatValues() const {
                return CatValues;
            }

            TConstArrayRef<TConstArrayRef<TTextValue>> GetTextValues() const {
                return TextValues;
            }

            TConstArrayRef<TVector<TEmbeddingValue>> GetEmbeddingValues() const {
                return EmbeddingValues;
            }

        private:
            const TObjectsDataProviderType& ObjectsData;

            TVector<IDynamicBlockIteratorPtr<TFloatValue>> FloatBlockIterators; // [repackedFlatIndex]
            TVector<IDynamicBlockIteratorPtr<TCatValue>> CatBlockIterators; // [repackedFlatIndex]
            TVector<IDynamicBlockIteratorPtr<TTextValue>> TextBlockIterators; // [repackedFlatIndex]
            TVector<IDynamicBlockIteratorPtr<TMaybeOwningConstArrayHolder<float>>> EmbeddingBlockIterators; // [repackedFlatIndex]

            TVector<TConstArrayRef<TFloatValue>> FloatValues; // [repackedFlatIndex][inBlockObjectIdx]
            TVector<TConstArrayRef<TCatValue>> CatValues; // [repackedFlatIndex][inBlockObjectIdx]
            TVector<TConstArrayRef<TTextValue>> TextValues; // [repackedFlatIndex][inBlockObjectIdx]
            TVector<TVector<TEmbeddingValue>> EmbeddingValues; // [repackedFlatIndex][inBlockObjectIdx]
        };

    }

    class TRawFeatureAccessor {
    public:
        TRawFeatureAccessor(const TRawFeaturesBlockIterator& rawFeaturesBlockIterator);

        Y_FORCE_INLINE auto GetFloatAccessor() const {
            return [this](TFeaturePosition position, size_t index) -> float {
                CB_ENSURE_INTERNAL(SafeIntegerCast<size_t>(position.FlatIndex) < FloatValues.size(),
                                   "position.FlatIndex " << position.FlatIndex << ", FloatValues.size() " << FloatValues.size());
                CB_ENSURE_INTERNAL(SafeIntegerCast<size_t>(index) < FloatValues[position.FlatIndex].size(),
                                   "index " << index << ", size " << FloatValues[position.FlatIndex].size());
                return FloatValues[position.FlatIndex][index];
            };
        }

        Y_FORCE_INLINE auto GetCatAccessor() const {
             return [this] (TFeaturePosition position, size_t index) -> ui32 {
                Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < CatValues.size());
                Y_ASSERT(SafeIntegerCast<size_t>(index) < CatValues[position.FlatIndex].size());
                return CatValues[position.FlatIndex][index];
            };
        };

        Y_FORCE_INLINE auto GetTextAccessor() const {
            return [this] (TFeaturePosition position, size_t index) -> TStringBuf {
                Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < TextValues.size());
                Y_ASSERT(SafeIntegerCast<size_t>(index) < TextValues[position.FlatIndex].size());
                return TextValues[position.FlatIndex][index];
            };
        }

        Y_FORCE_INLINE auto GetEmbeddingAccessor() const {
            return [this] (TFeaturePosition position, size_t index) -> TConstArrayRef<float> {
                Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < EmbeddingValues.size());
                Y_ASSERT(SafeIntegerCast<size_t>(index) < EmbeddingValues[position.FlatIndex].size());
                return EmbeddingValues[position.FlatIndex][index];
            };
        }
    private:
        TConstArrayRef<TConstArrayRef<float>> FloatValues; // [repackedFlatIndex][inBlockObjectIdx]
        TConstArrayRef<TConstArrayRef<ui32>> CatValues;  // [repackedFlatIndex][inBlockObjectIdx]
        TConstArrayRef<TConstArrayRef<TString>> TextValues; // [repackedFlatIndex][inBlockObjectIdx]
        TConstArrayRef<TVector<TConstArrayRef<float>>> EmbeddingValues; // [repackedFlatIndex][inBlockObjectIdx]
    };


    class TRawFeaturesBlockIterator
        : public NDetail::TFeaturesBlockIteratorBase<TRawObjectsDataProvider, float, ui32, TString,
                                                     TConstArrayRef<float>, TRawFeatureAccessor>
    {
    public:
        using TBase
            = NDetail::TFeaturesBlockIteratorBase<TRawObjectsDataProvider, float, ui32, TString,
                                                  TConstArrayRef<float>, TRawFeatureAccessor>;

    public:
        TRawFeaturesBlockIterator(
            const TFullModel& model,
            const TRawObjectsDataProvider& objectsData,
            const THashMap<ui32, ui32>& columnReorderMap,
            ui32 objectOffset)
            : TBase(model, objectsData, columnReorderMap, objectOffset)
        {}

        void Accept(IFeaturesBlockIteratorVisitor* visitor) const override {
            visitor->Visit(*this);
        }

        TRawFeatureAccessor GetAccessor() const {
            return TRawFeatureAccessor(*this);
        }
    };


    inline TRawFeatureAccessor::TRawFeatureAccessor(const TRawFeaturesBlockIterator& rawFeaturesBlockIterator)
        : FloatValues(rawFeaturesBlockIterator.GetFloatValues())
        , CatValues(rawFeaturesBlockIterator.GetCatValues())
        , TextValues(rawFeaturesBlockIterator.GetTextValues())
        , EmbeddingValues(rawFeaturesBlockIterator.GetEmbeddingValues())
    {}


    class TQuantizedFeatureAccessor {
    public:
        TQuantizedFeatureAccessor(const TQuantizedFeaturesBlockIterator& quantizedFeaturesBlockIterator);

        Y_FORCE_INLINE auto GetFloatAccessor() const {
            return [this](TFeaturePosition position, size_t index) -> ui8 {
                Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < FloatValues.size());
                Y_ASSERT(SafeIntegerCast<size_t>(index) < FloatValues[position.FlatIndex].size());
                return FloatBinsRemap[position.FlatIndex][FloatValues[position.FlatIndex][index]];
            };
        }

        // returns hashed cat value
        Y_FORCE_INLINE auto GetCatAccessor() const {
            return [this](TFeaturePosition position, size_t index) -> ui32 {
                Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < CatValues.size());
                Y_ASSERT(SafeIntegerCast<size_t>(index) < CatValues[position.FlatIndex].size());
                return CatFeatureBinToHashedValueRemap[position.FlatIndex][CatValues[position.FlatIndex][index]];
            };
        };
    private:
        TConstArrayRef<TConstArrayRef<ui8>> FloatValues; // [repackedFlatIndex][inBlockObjectIdx]
        TConstArrayRef<TConstArrayRef<ui8>> FloatBinsRemap; // [repackedFlatIndex][binInData]

        TConstArrayRef<TConstArrayRef<ui32>> CatValues; // [repackedFlatIndex][inBlockObjectIdx]
        TConstArrayRef<TConstArrayRef<ui32>> CatFeatureBinToHashedValueRemap; // [repackedFlatIndex][binInData]
    };

    class TQuantizedFeaturesBlockIterator
        : public NDetail::TFeaturesBlockIteratorBase<
            TQuantizedObjectsDataProvider,
            ui8,
            ui32,
            NCB::TText,
            TConstArrayRef<float>,
            TQuantizedFeatureAccessor>
    {
    public:
        using TBase = TFeaturesBlockIteratorBase<
            TQuantizedObjectsDataProvider,
            ui8,
            ui32,
            NCB::TText,
            TConstArrayRef<float>,
            TQuantizedFeatureAccessor>;

    public:
        TQuantizedFeaturesBlockIterator(
            const TFullModel& model,
            const TQuantizedObjectsDataProvider& objectsData,
            const THashMap<ui32, ui32>& columnReorderMap,
            ui32 objectOffset)
            : TBase(model, objectsData, columnReorderMap, objectOffset)
            , FloatBinsRemap(
                GetFloatFeaturesBordersRemap(
                    model,
                    columnReorderMap,
                    *objectsData.GetQuantizedFeaturesInfo()
                )
              )
            , FloatBinsRemapRef(FloatBinsRemap.begin(), FloatBinsRemap.end())
            , CatFeatureBinToHashedValueRemap(
                GetCatFeaturesBinToHashedValueRemap(
                    model,
                    columnReorderMap,
                    *objectsData.GetQuantizedFeaturesInfo()
                )
              )
            , CatFeatureBinToHashedValueRemapRef(
                CatFeatureBinToHashedValueRemap.begin(),
                CatFeatureBinToHashedValueRemap.end()
              )
        {}

        void Accept(IFeaturesBlockIteratorVisitor* visitor) const override {
            visitor->Visit(*this);
        }

        TQuantizedFeatureAccessor GetAccessor() const {
            return TQuantizedFeatureAccessor(*this);
        }

        TConstArrayRef<TConstArrayRef<ui8>> GetFloatBinsRemap() const {
            return FloatBinsRemapRef;
        }

        TConstArrayRef<TConstArrayRef<ui32>> GetCatFeatureBinToHashedValueRemap() const {
            return CatFeatureBinToHashedValueRemapRef;
        }

    private:
        TVector<TVector<ui8>> FloatBinsRemap;
        TVector<TConstArrayRef<ui8>> FloatBinsRemapRef;

        TVector<TVector<ui32>> CatFeatureBinToHashedValueRemap;
        TVector<TConstArrayRef<ui32>> CatFeatureBinToHashedValueRemapRef;
    };


    inline TQuantizedFeatureAccessor::TQuantizedFeatureAccessor(
        const TQuantizedFeaturesBlockIterator& quantizedFeaturesBlockIterator)
        : FloatValues(quantizedFeaturesBlockIterator.GetFloatValues())
        , FloatBinsRemap(quantizedFeaturesBlockIterator.GetFloatBinsRemap())
        , CatValues(quantizedFeaturesBlockIterator.GetCatValues())
        , CatFeatureBinToHashedValueRemap(quantizedFeaturesBlockIterator.GetCatFeatureBinToHashedValueRemap())
    {}


    THolder<IFeaturesBlockIterator> CreateFeaturesBlockIterator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        size_t start,
        size_t end);

}
