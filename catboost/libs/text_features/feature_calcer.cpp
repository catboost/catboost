#include "feature_calcer.h"

#include "helpers.h"

namespace NCB {
    static void WriteMagic(const char* magic, ui32 magicSize, ui32 alignment, TCountingOutput* stream) {
        stream->Write(magic, magicSize);
        AddPadding(stream, alignment);
        Y_ASSERT(stream->Counter() % alignment == 0);
    }

    static void ReadMagic(const char* magic, ui32 magicSize, ui32 alignment, TCountingInput* stream) {
        Y_UNUSED(magic);
        TArrayHolder<char> loadedMagic = new char[magicSize];
        ui32 loadedBytes = stream->Load(loadedMagic.Get(), magicSize);
        CB_ENSURE(
            loadedBytes == magicSize && Equal(loadedMagic.Get(), loadedMagic.Get() + magicSize, magic),
            "Failed to deserialize calcer: couldn't read magic"
        );
        SkipPadding(stream, alignment);
    }

    void TTextCalcerSerializer::Save(IOutputStream* s, const TTextFeatureCalcer& calcer) {
        TCountingOutput stream(s);
        WriteMagic(CalcerMagic.data(), MagicSize, Alignment, &stream);

        ::Save(&stream, static_cast<ui32>(calcer.Type()));
        ::Save(&stream, calcer);
    }

    TTextFeatureCalcerPtr TTextCalcerSerializer::Load(IInputStream* s) {
        TCountingInput stream(s);
        ReadMagic(CalcerMagic.data(), MagicSize, Alignment, &stream);

        static_assert(sizeof(EFeatureCalcerType) == sizeof(ui32));
        EFeatureCalcerType calcerType;
        ::Load(&stream, calcerType);

        TTextFeatureCalcer* calcer = TTextFeatureCalcerFactory::Construct(calcerType);
        ::Load(&stream, *calcer);

        return calcer;
    }

    void TTextFeatureCalcer::Save(IOutputStream* stream) const {
        flatbuffers::FlatBufferBuilder builder;
        auto calcerFbs = SaveParametersToFB(builder);
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

        auto calcer = flatbuffers::GetRoot<NCatBoostFbs::TFeatureCalcer>(buffer.Get());
        LoadParametersFromFB(calcer);

        LoadLargeParameters(stream);
    }

    flatbuffers::Offset<NCatBoostFbs::TFeatureCalcer> TTextFeatureCalcer::SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const {
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
