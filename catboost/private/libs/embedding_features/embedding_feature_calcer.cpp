#include "embedding_feature_calcer.h"

namespace NCB {
    void TEmbeddingFeatureCalcer::TrimFeatures(TConstArrayRef<ui32> featureIndices) {
        const ui32 featureCount = FeatureCount();
        CB_ENSURE(
            featureIndices.size() <= featureCount && featureIndices.back() < featureCount,
            "Specified trim feature indices is greater than number of features that calcer produce"
        );
        ActiveFeatureIndices = TVector<ui32>(featureIndices.begin(), featureIndices.end());
    }

    TConstArrayRef<ui32> TEmbeddingFeatureCalcer::GetActiveFeatureIndices() const {
        return MakeConstArrayRef(ActiveFeatureIndices);
    }

    void TEmbeddingFeatureCalcer::Save(IOutputStream* stream) const {
        flatbuffers::FlatBufferBuilder builder;
        TEmbeddingCalcerFbs anyCalcerFbs = SaveParametersToFB(builder);
        auto fbsGuid = CreateFbsGuid(Guid);

        auto calcerFbs = NCatBoostFbs::NEmbeddings::CreateTEmbeddingCalcer(
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

    void TEmbeddingFeatureCalcer::Load(IInputStream* stream) {
        ui64 bufferSize;
        ::Load(stream, bufferSize);
        TArrayHolder<ui8> buffer = TArrayHolder<ui8>(new ui8[bufferSize]);
        const ui64 loadedBytes = stream->Load(buffer.Get(), bufferSize);
        CB_ENSURE(loadedBytes == bufferSize, "Failed to deserialize: Couldn't read calcer flatbuffer");

        {
            flatbuffers::Verifier verifier{buffer.Get(), static_cast<size_t>(bufferSize)};
            CB_ENSURE(
                NCatBoostFbs::NEmbeddings::VerifyTEmbeddingCalcerBuffer(verifier),
                "Flatbuffers model verification failed"
            );
        }

        auto calcer = flatbuffers::GetRoot<NCatBoostFbs::NEmbeddings::TEmbeddingCalcer>(buffer.Get());
        ActiveFeatureIndices = TVector<ui32>(
            calcer->ActiveFeatureIndices()->begin(),
            calcer->ActiveFeatureIndices()->end()
        );
        const TGuid guid = GuidFromFbs(calcer->Id());
        SetId(guid);
        LoadParametersFromFB(calcer);
        LoadLargeParameters(stream);
    }

    TEmbeddingFeatureCalcer::TEmbeddingCalcerFbs TEmbeddingFeatureCalcer::SaveParametersToFB(flatbuffers::FlatBufferBuilder&) const {
        CB_ENSURE(false, "Serialization to flatbuffer is not implemented");
    }

    void TEmbeddingFeatureCalcer::SaveLargeParameters(IOutputStream*) const {
        CB_ENSURE(false, "Serialization is not implemented");
    }

    void TEmbeddingFeatureCalcer::LoadParametersFromFB(const NCatBoostFbs::NEmbeddings::TEmbeddingCalcer*) {
        CB_ENSURE(false, "Deserialization from flatbuffer is not implemented");
    }

    void TEmbeddingFeatureCalcer::LoadLargeParameters(IInputStream*) {
        CB_ENSURE(false, "Deserialization is not implemented");
    }

    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> TEmbeddingFeatureCalcer::ActiveFeatureIndicesToFB(flatbuffers::FlatBufferBuilder& builder) const {
        return builder.CreateVector(
            reinterpret_cast<const uint32_t*>(ActiveFeatureIndices.data()),
            ActiveFeatureIndices.size()
        );
    }

    void TEmbeddingCalcerSerializer::Save(IOutputStream* stream, const TEmbeddingFeatureCalcer& calcer) {
        WriteMagic(CalcerMagic.data(), MagicSize, Alignment, stream);

        ::Save(stream, static_cast<ui32>(calcer.Type()));
        ::Save(stream, calcer);
    }

    TEmbeddingFeatureCalcerPtr TEmbeddingCalcerSerializer::Load(IInputStream* stream) {
        ReadMagic(CalcerMagic.data(), MagicSize, Alignment, stream);

        static_assert(sizeof(EFeatureCalcerType) == sizeof(ui32));
        EFeatureCalcerType calcerType;
        ::Load(stream, calcerType);

        TEmbeddingFeatureCalcer* calcer = TEmbeddingFeatureCalcerFactory::Construct(calcerType);
        ::Load(stream, *calcer);

        return calcer;
    }

}
