#pragma once

#include "helpers.h"

#include <catboost/private/libs/data_types/text.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/helpers/guid.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/flatbuffers/guid.fbs.h>
#include <catboost/private/libs/text_features/flatbuffers/feature_calcers.fbs.h>

#include <library/cpp/object_factory/object_factory.h>
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
            CB_ENSURE(false, "Serialization not allowed");
        };

        virtual void Load(IInputStream*) {
            CB_ENSURE(false, "Deserialization not allowed");
        };

        virtual void TrimFeatures(TConstArrayRef<ui32> featureIndices) = 0;

        virtual TGuid Id() const = 0;
    };

    class TTextCalcerSerializer;

    class TOutputFloatIterator {
    public:
        inline TOutputFloatIterator(float* data, ui64 size)
            : DataPtr(data)
            , EndPtr(data + size)
            , Step(1) {}

        inline TOutputFloatIterator(float* data, ui64 step, ui64 size)
            : DataPtr(data)
            , EndPtr(data + size)
            , Step(step) {}

        inline float& operator*() {
            Y_ASSERT(IsValid());
            return *DataPtr;
        }
        inline TOutputFloatIterator& operator++() {
            Y_ASSERT(IsValid());
            DataPtr += Step;
            return *this;
        }
        inline const TOutputFloatIterator operator++(int) {
            TOutputFloatIterator tmp(*this);
            operator++();
            return tmp;
        }
        inline bool IsValid() const {
            return DataPtr < EndPtr;
        }

    private:
        float* DataPtr;
        float* EndPtr;
        ui64 Step;
    };

    class TTextFeatureCalcer : public IFeatureCalcer {
    public:
        TTextFeatureCalcer(ui32 baseFeatureCount, const TGuid& calcerId)
            : ActiveFeatureIndices(baseFeatureCount)
            , Guid(calcerId)
        {
            Iota(ActiveFeatureIndices.begin(), ActiveFeatureIndices.end(), 0);
        }

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

        TGuid Id() const override {
            return Guid;
        }

        void SetId(const TGuid& guid) {
            Guid = guid;
        }

    protected:
        class TFeatureCalcerFbs {
        public:
            using TCalcerFbsImpl = flatbuffers::Offset<void>;
            using ECalcerFbsType = NCatBoostFbs::TAnyFeatureCalcer;

            TFeatureCalcerFbs(
                ECalcerFbsType calcerFbsType,
                TCalcerFbsImpl featureCalcerFbs
            )
                : CalcerType(calcerFbsType)
                , CalcerFlatBuffer(featureCalcerFbs)
            {}

            ECalcerFbsType GetCalcerType() const {
                return CalcerType;
            }

            const TCalcerFbsImpl& GetCalcerFlatBuffer() const {
                return CalcerFlatBuffer;
            }

        private:
            ECalcerFbsType CalcerType;
            TCalcerFbsImpl CalcerFlatBuffer;
        };

        template <class F>
        inline void ForEachActiveFeature(F&& func) const {
            for (ui32 featureId: GetActiveFeatureIndices()) {
                func(featureId);
            }
        }

        NCatBoostFbs::TGuid GetFbsGuid() const {
            return CreateFbsGuid(Guid);
        }

        virtual TFeatureCalcerFbs SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const;
        virtual void SaveLargeParameters(IOutputStream*) const;

        virtual void LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer*);
        virtual void LoadLargeParameters(IInputStream*);

        flatbuffers::Offset<flatbuffers::Vector<uint32_t>> ActiveFeatureIndicesToFB(flatbuffers::FlatBufferBuilder& builder) const;

    private:
        TVector<ui32> ActiveFeatureIndices;
        TGuid Guid = CreateGuid();
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
