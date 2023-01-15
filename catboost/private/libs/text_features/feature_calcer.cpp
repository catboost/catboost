#include "feature_calcer.h"

#include <catboost/libs/helpers/serialization.h>

#include <util/system/guard.h>

namespace NCB {
    void TTextCalcerSerializer::Save(IOutputStream* stream, const TTextFeatureCalcer& calcer) {
        WriteMagic(CalcerMagic.data(), MagicSize, Alignment, stream);

        ::Save(stream, static_cast<ui32>(calcer.Type()));
        ::Save(stream, calcer);
    }

    TTextFeatureCalcerPtr TTextCalcerSerializer::Load(IInputStream* stream) {
        ReadMagic(CalcerMagic.data(), MagicSize, Alignment, stream);

        static_assert(sizeof(EFeatureCalcerType) == sizeof(ui32));
        EFeatureCalcerType calcerType;
        ::Load(stream, calcerType);

        TTextFeatureCalcer* calcer = TTextFeatureCalcerFactory::Construct(calcerType);
        ::Load(stream, *calcer);

        return calcer;
    }

    void TTextFeatureCalcer::Save(IOutputStream* stream) const {
        flatbuffers::FlatBufferBuilder builder;
        TFeatureCalcerFbs anyCalcerFbs = SaveParametersToFB(builder);
        auto fbsGuid = CreateFbsGuid(Guid);

        auto calcerFbs = NCatBoostFbs::CreateTFeatureCalcer(
            builder,
            &fbsGuid,
            ActiveFeatureIndicesToFB(builder),
            anyCalcerFbs.GetCalcerType(),
            anyCalcerFbs.GetCalcerFlatBuffer()
        );
        builder.Finish(calcerFbs);

        {
            ui64 bufferSize = static_cast<ui64>(builder.GetSize());
            ::Save(stream, bufferSize);
            stream->Write(builder.GetBufferPointer(), bufferSize);
        }

        SaveLargeParameters(stream);
    }

    void TTextFeatureCalcer::Load(IInputStream* stream) {
        ui64 bufferSize;
        ::Load(stream, bufferSize);
        TArrayHolder<ui8> buffer = new ui8[bufferSize];
        const ui64 loadedBytes = stream->Load(buffer.Get(), bufferSize);
        CB_ENSURE(loadedBytes == bufferSize, "Failed to deserialize: Couldn't read calcer flatbuffer");

        {
            flatbuffers::Verifier verifier{buffer.Get(), static_cast<size_t>(bufferSize)};
            CB_ENSURE(
                NCatBoostFbs::VerifyTFeatureCalcerBuffer(verifier),
                "Flatbuffers model verification failed"
            );
        }

        auto calcer = flatbuffers::GetRoot<NCatBoostFbs::TFeatureCalcer>(buffer.Get());
        ActiveFeatureIndices = TVector<ui32>(
            calcer->ActiveFeatureIndices()->begin(),
            calcer->ActiveFeatureIndices()->end()
        );
        const TGuid guid = GuidFromFbs(calcer->Id());
        SetId(guid);

        LoadParametersFromFB(calcer);

        LoadLargeParameters(stream);
    }

    TTextFeatureCalcer::TFeatureCalcerFbs TTextFeatureCalcer::SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const {
        Y_FAIL("Serialization to flatbuffer is not implemented");
    }

    void TTextFeatureCalcer::SaveLargeParameters(IOutputStream*) const {
        Y_FAIL("Serialization is not implemented");
    }

    void TTextFeatureCalcer::LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer*) {
        Y_FAIL("Deserialization from flatbuffer is not implemented");
    }

    void TTextFeatureCalcer::LoadLargeParameters(IInputStream*) {
        Y_FAIL("Deserialization is not implemented");
    }

    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> TTextFeatureCalcer::ActiveFeatureIndicesToFB(flatbuffers::FlatBufferBuilder& builder) const {
        return builder.CreateVector(
            reinterpret_cast<const uint32_t*>(ActiveFeatureIndices.data()),
            ActiveFeatureIndices.size()
        );
    }

    void TTextFeatureCalcer::TrimFeatures(TConstArrayRef<ui32> featureIndices) {
        const ui32 featureCount = FeatureCount();
        CB_ENSURE(
            featureIndices.size() <= featureCount && featureIndices.back() < featureCount,
            "Specified trim feature indices is greater than number of features that calcer produce"
        );
        ActiveFeatureIndices = TVector<ui32>(featureIndices.begin(), featureIndices.end());
    }

    ui32 TTextFeatureCalcer::FeatureCount() const {
        return GetActiveFeatureIndices().size();
    }

    TConstArrayRef<ui32> TTextFeatureCalcer::GetActiveFeatureIndices() const {
        return MakeConstArrayRef(ActiveFeatureIndices);
    }

    TOutputFloatIterator::TOutputFloatIterator(float* data, ui64 size)
        : DataPtr(data)
        , EndPtr(data + size)
        , Step(1) {}

    TOutputFloatIterator::TOutputFloatIterator(float* data, ui64 step, ui64 size)
        : DataPtr(data)
        , EndPtr(data + size)
        , Step(step) {}

    const TOutputFloatIterator TOutputFloatIterator::operator++(int) {
        TOutputFloatIterator tmp(*this);
        operator++();
        return tmp;
    }

    bool TOutputFloatIterator::IsValid() {
        return DataPtr < EndPtr;
    }

    TOutputFloatIterator& TOutputFloatIterator::operator++() {
        Y_ASSERT(IsValid());
        DataPtr += Step;
        return *this;
    }

    float& TOutputFloatIterator::operator*() {
        Y_ASSERT(IsValid());
        return *DataPtr;
    }

}
