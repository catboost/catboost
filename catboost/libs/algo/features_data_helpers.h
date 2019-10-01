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

        template <class TObjectsDataProviderType, class TFloatValue, class TCatValue, class TAccessor>
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
                size_t flatFeatureCount = model.ObliviousTrees->GetFlatFeatureVectorExpectedSize();
                FloatBlockIterators.resize(flatFeatureCount);
                CatBlockIterators.resize(flatFeatureCount);
                FloatValues.resize(flatFeatureCount);
                CatValues.resize(flatFeatureCount);

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
                    FloatBlockIterators[modelFlatFeatureIdx]
                        = (*ObjectsData.GetFloatFeature(internalFeatureIdx))
                            ->GetBlockIterator(objectOffset);
                } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                    CatBlockIterators[modelFlatFeatureIdx]
                        = (*ObjectsData.GetCatFeature(internalFeatureIdx))
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

        private:
            const TObjectsDataProviderType& ObjectsData;

            TVector<IDynamicBlockIteratorPtr<TFloatValue>> FloatBlockIterators; // [repackedFlatIndex]
            TVector<IDynamicBlockIteratorPtr<TCatValue>> CatBlockIterators; // [repackedFlatIndex]

            TVector<TConstArrayRef<TFloatValue>> FloatValues; // [repackedFlatIndex][inBlockObjectIdx]
            TVector<TConstArrayRef<TCatValue>> CatValues; // [repackedFlatIndex][inBlockObjectIdx]
        };

    }

    class TRawFeatureAccessor {
    public:
        TRawFeatureAccessor(const TRawFeaturesBlockIterator& rawFeaturesBlockIterator);

        Y_FORCE_INLINE auto GetFloatAccessor() const {
            return [this](TFeaturePosition position, size_t index) -> float {
                Y_ASSERT(SafeIntegerCast<size_t>(position.FlatIndex) < FloatValues.size());
                Y_ASSERT(SafeIntegerCast<size_t>(index) < FloatValues[position.FlatIndex].size());
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
    private:
        TConstArrayRef<TConstArrayRef<float>> FloatValues; // [repackedFlatIndex][inBlockObjectIdx]
        TConstArrayRef<TConstArrayRef<ui32>> CatValues;  // [repackedFlatIndex][inBlockObjectIdx]
    };


    class TRawFeaturesBlockIterator
        : public NDetail::TFeaturesBlockIteratorBase<TRawObjectsDataProvider, float, ui32, TRawFeatureAccessor>
    {
    public:
        using TBase
            = NDetail::TFeaturesBlockIteratorBase<TRawObjectsDataProvider, float, ui32, TRawFeatureAccessor>;

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

        Y_FORCE_INLINE auto GetCatAccessor() const {
             return [] (TFeaturePosition , size_t ) -> ui32 {
                 Y_FAIL();
                 return 0;
            };
        };
    private:
        TConstArrayRef<TConstArrayRef<ui8>> FloatValues; // [repackedFlatIndex][inBlockObjectIdx]
        TConstArrayRef<TConstArrayRef<ui8>> FloatBinsRemap; // [repackedFlatIndex][binInData]
    };


    class TQuantizedFeaturesBlockIterator
        : public NDetail::TFeaturesBlockIteratorBase<
            TQuantizedObjectsDataProvider,
            ui8,
            ui32,
            TQuantizedFeatureAccessor>
    {
    public:
        using TBase = TFeaturesBlockIteratorBase<
            TQuantizedObjectsDataProvider,
            ui8,
            ui32,
            TQuantizedFeatureAccessor>;

    public:
        TQuantizedFeaturesBlockIterator(
            const TFullModel& model,
            const TQuantizedObjectsDataProvider& objectsData,
            const THashMap<ui32, ui32>& columnReorderMap,
            ui32 objectOffset)
            : TBase(model, objectsData, columnReorderMap, objectOffset)
            , FloatBinsRemap(GetFloatFeaturesBordersRemap(model, *objectsData.GetQuantizedFeaturesInfo()))
            , FloatBinsRemapRef(FloatBinsRemap.begin(), FloatBinsRemap.end())
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

    private:
        TVector<TVector<ui8>> FloatBinsRemap;
        TVector<TConstArrayRef<ui8>> FloatBinsRemapRef;
    };


    inline TQuantizedFeatureAccessor::TQuantizedFeatureAccessor(
        const TQuantizedFeaturesBlockIterator& quantizedFeaturesBlockIterator)
        : FloatValues(quantizedFeaturesBlockIterator.GetFloatValues())
        , FloatBinsRemap(quantizedFeaturesBlockIterator.GetFloatBinsRemap())
    {}


    THolder<IFeaturesBlockIterator> CreateFeaturesBlockIterator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        size_t start,
        size_t end);

}
