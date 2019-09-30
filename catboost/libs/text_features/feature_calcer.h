#pragma once

#include <catboost/libs/data_types/text.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/helpers/guid.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/text_features/flatbuffers/feature_calcers.fbs.h>

#include <library/object_factory/object_factory.h>
#include <util/generic/ptr.h>
#include <util/generic/guid.h>
#include <util/stream/input.h>
#include <util/system/mutex.h>

#include <contrib/libs/cxxsupp/libcxx/include/array>

namespace NCB {

    class IFeatureCalcer : public TThrRefBase {
    public:

        virtual EFeatureCalcerType Type() const = 0;

        virtual ui32 FeatureCount() const = 0;

        //TODO: (noxoomo, kirillovs@): remove duplication with ICtrProvider
        virtual bool IsSerializable() const {
            return false;
        }

        virtual void Save(IOutputStream*) const {
            Y_FAIL("Serialization not allowed");
        };

        virtual void Load(IInputStream*) {
            Y_FAIL("Deserialization not allowed");
        };

        virtual void TrimFeatures(TConstArrayRef<ui32> featureIndices) = 0;
    };

    class TTextCalcerSerializer;

    class TOutputFloatIterator {
    public:
        TOutputFloatIterator(float* data, ui64 size);
        TOutputFloatIterator(float* data, ui64 step, ui64 size);

        float& operator*();
        TOutputFloatIterator& operator++();
        const TOutputFloatIterator operator++(int);

        bool IsValid();

    private:
        float* DataPtr;
        float* EndPtr;
        ui64 Step;
    };

    class TTextFeatureCalcer : public IFeatureCalcer {
    public:
        virtual void Compute(const TText& text, TOutputFloatIterator outputFeaturesIterator) const = 0;

        void Compute(const TText& text, TArrayRef<float> result) const {
            Compute(text, TOutputFloatIterator(result.begin(), result.size()));
        }

        TVector<float> Compute(const TText& text) const {
            TVector<float> result;
            result.yresize(FeatureCount());
            Compute(text, TOutputFloatIterator(result.data(), result.size()));
            return result;
        }

        void Save(IOutputStream* stream) const final;
        void Load(IInputStream* stream) final;

        void TrimFeatures(TConstArrayRef<ui32> featureIndices) override;
        TConstArrayRef<ui32> GetActiveFeatureIndices() const;
        ui32 FeatureCount() const override;

    protected:
        virtual ui32 BaseFeatureCount() const = 0;

        template <class F>
        void ForEachActiveFeature(F&& func) const {
            for (ui32 featureId: GetActiveFeatureIndices()) {
                func(featureId);
            }
        }

        virtual flatbuffers::Offset<NCatBoostFbs::TFeatureCalcer> SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const;
        virtual void SaveLargeParameters(IOutputStream*) const;

        virtual void LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer*);
        virtual void LoadLargeParameters(IInputStream*);

        flatbuffers::Offset<flatbuffers::Vector<uint32_t>> ActiveFeatureIndicesToFB(flatbuffers::FlatBufferBuilder& builder) const;

    private:
        mutable TVector<ui32> ActiveFeatureIndices;
        TMutex InitActiveFeatureIndicesLock;
    };

    using TTextFeatureCalcerPtr = TIntrusivePtr<TTextFeatureCalcer>;
    using TTextFeatureCalcerFactory = NObjectFactory::TParametrizedObjectFactory<TTextFeatureCalcer,
                                                                                 EFeatureCalcerType>;

    class ITextCalcerVisitor : public TThrRefBase {
    public:
        virtual void Update(ui32 classId, const TText& text, TTextFeatureCalcer* featureCalcer) = 0;
    };

    using TTextCalcerVisitorPtr = TIntrusivePtr<ITextCalcerVisitor>;

    class TTextCalcerSerializer {
    public:
        static void Save(IOutputStream* stream, const TTextFeatureCalcer& calcer);
        static TTextFeatureCalcerPtr Load(IInputStream* stream);

    private:
        static constexpr std::array<char, 16> CalcerMagic = {"FeatureCalcerV1"};
        static constexpr ui32 MagicSize = CalcerMagic.size();
        static constexpr ui32 Alignment = 16;
    };
}
