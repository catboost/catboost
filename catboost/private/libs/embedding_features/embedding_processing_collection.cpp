#include "embedding_processing_collection.h"

#include <catboost/libs/helpers/serialization.h>
#include <catboost/private/libs/embedding_features/flatbuffers/embedding_processing_collection.fbs.h>



namespace NCB {

    // TODO(oganes): move copypaste to proper place

    template <class T>
    static bool TryLoad(IInputStream* stream, T& value) {
        const ui32 readLen = stream->Load(&value, sizeof(T));
        CB_ENSURE_INTERNAL(
            readLen == 0 || readLen == sizeof(T),
            "Failed to deserialize: only half of header was read"
        );
        return readLen == sizeof(T);
    }

    static decltype(auto) FBSerializeGuidArray(
        flatbuffers::FlatBufferBuilder& builder,
        const TVector<TGuid>& guids
    ) {
        TVector<NCatBoostFbs::TGuid> fbGuids;
        fbGuids.reserve(guids.size());
        for (const auto& guid: guids) {
            auto fbGuid = CreateFbsGuid(guid);
            fbGuids.push_back(fbGuid);
        }

        return builder.CreateVectorOfStructs(fbGuids.data(), fbGuids.size());
    }

    static void FBDeserializeGuidArray(
        const flatbuffers::Vector<const NCatBoostFbs::TGuid*>& fbGuids,
        TVector<TGuid>* guids
    ) {
        guids->clear();
        guids->reserve(fbGuids.size());

        for (auto fbGuid : fbGuids) {
            guids->push_back(GuidFromFbs(fbGuid));
        }
    }

    static decltype(auto) FBSerializeAdjacencyList(
        flatbuffers::FlatBufferBuilder& builder,
        const TVector<TVector<ui32>>& bipartiteGraph
    ) {
        using namespace flatbuffers;
        using namespace NCatBoostFbs::NEmbeddings;

        TVector<Offset<AdjacencyList>> fbBipartiteGraph;
        fbBipartiteGraph.reserve(bipartiteGraph.size());

        for (const auto& adjacencyList: bipartiteGraph) {
            fbBipartiteGraph.push_back(CreateAdjacencyListDirect(builder, &adjacencyList));
        }

        return builder.CreateVector(fbBipartiteGraph.data(), fbBipartiteGraph.size());
    }

    static void FBDeserializeAdjacencyList(
        const flatbuffers::Vector<flatbuffers::Offset<NCatBoostFbs::NEmbeddings::AdjacencyList>>& fbBipartiteGraph,
        TVector<TVector<ui32>>* bipartiteGraph
    ) {
        bipartiteGraph->clear();
        bipartiteGraph->reserve(fbBipartiteGraph.size());

        for (auto fbAdjacencyList : fbBipartiteGraph) {
            const flatbuffers::Vector<uint32_t>* ids = fbAdjacencyList->Ids();
            bipartiteGraph->emplace_back(ids->begin(), ids->end());
        }
    }

    TEmbeddingProcessingCollection::TEmbeddingProcessingCollection(
        TVector<TEmbeddingFeatureCalcerPtr> calcers,
        TVector<TVector<ui32>> perEmbeddingFeatureCalcers
    )
        : FeatureCalcers(std::move(calcers))
        , PerEmbeddingFeatureCalcers(std::move(perEmbeddingFeatureCalcers))
    {
        FeatureCalcerId.resize(FeatureCalcers.size());
        for (ui32 idx : xrange(FeatureCalcers.size())) {
            FeatureCalcerId[idx] = FeatureCalcers[idx]->Id();
        }

        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

    void TEmbeddingProcessingCollection::CalcRuntimeData() {
        ui32 currentOffset = 0;
        for (ui32 embeddingFeatureIdx: xrange(PerEmbeddingFeatureCalcers.size())) {
            for (ui32 calcerId: PerEmbeddingFeatureCalcers[embeddingFeatureIdx]) {
                FeatureCalcerOffset[calcerId] = currentOffset;
                currentOffset += FeatureCalcers[calcerId]->FeatureCount();
            }
        }
        for (ui32 calcerFlatIdx: xrange(FeatureCalcerId.size())) {
            CalcerGuidToFlatIdx[FeatureCalcerId[calcerFlatIdx]] = calcerFlatIdx;
        }
    }

    void TEmbeddingProcessingCollection::CheckPerFeatureIdx() const {
        for (ui32 embeddingFeatureId: xrange(PerEmbeddingFeatureCalcers.size())) {
            for (ui32 calcerId: PerEmbeddingFeatureCalcers[embeddingFeatureId]) {
                CB_ENSURE(
                    calcerId < FeatureCalcers.size(),
                    "For embedding feature id=" << embeddingFeatureId << " specified feature calcer id="
                        << calcerId << " which is greater than number of calcers"
                );
            }
        }
    }

    ui32 TEmbeddingProcessingCollection::GetAbsoluteCalcerOffset(ui32 calcerIdx) const {
        return FeatureCalcerOffset.at(calcerIdx);
    }

    ui32 TEmbeddingProcessingCollection::GetRelativeCalcerOffset(ui32 embeddingFeatureIdx, ui32 calcerIdx) const {
        return (GetAbsoluteCalcerOffset(calcerIdx) -
                GetAbsoluteCalcerOffset(PerEmbeddingFeatureCalcers[embeddingFeatureIdx][0]));
    }

    ui32 TEmbeddingProcessingCollection::GetAbsoluteCalcerOffset(const TGuid& calcerGuid) const {
        CB_ENSURE(
            CalcerGuidToFlatIdx.contains(calcerGuid),
            "There is no calcer with " << LabeledOutput(calcerGuid)
        );
        return GetAbsoluteCalcerOffset(CalcerGuidToFlatIdx.at(calcerGuid));
    }

    ui32 TEmbeddingProcessingCollection::GetRelativeCalcerOffset(ui32 textFeatureIdx, const TGuid& calcerGuid) const {
        return GetRelativeCalcerOffset(textFeatureIdx, CalcerGuidToFlatIdx.at(calcerGuid));
    }

    ui32 TEmbeddingProcessingCollection::NumberOfOutputFeatures(ui32 featureId) const {
        ui32 sum = 0;
        for (const auto& calcerId: PerEmbeddingFeatureCalcers[featureId]) {
            sum += FeatureCalcers[calcerId]->FeatureCount();
        }
        return sum;
    }

    bool TEmbeddingProcessingCollection::operator==(const TEmbeddingProcessingCollection& rhs) {
        return std::tie(
            FeatureCalcerId,
            PerEmbeddingFeatureCalcers
        ) == std::tie(
            rhs.FeatureCalcerId,
            rhs.PerEmbeddingFeatureCalcers
        );
    }

    bool TEmbeddingProcessingCollection::operator!=(const TEmbeddingProcessingCollection& rhs) {
        return !(*this == rhs);
    }

    ui32 TEmbeddingProcessingCollection::TotalNumberOfOutputFeatures() const {
        ui32 sum = 0;
        for (const auto& embeddingCalcers: PerEmbeddingFeatureCalcers) {
            for (ui32 calcerId: embeddingCalcers) {
                sum += FeatureCalcers[calcerId]->FeatureCount();
            }
        }
        return sum;
    }

    void TEmbeddingProcessingCollection::CalcFeatures(
        TConstArrayRef<TEmbeddingsArray> embeddingFeature,
        ui32 embeddingFeatureIdx,
        TArrayRef<float> result
    )  const {
        size_t docCount = embeddingFeature.size();
        for (ui32 calcerId: PerEmbeddingFeatureCalcers[embeddingFeatureIdx]) {
            const auto& calcer = FeatureCalcers[calcerId];

            const size_t calcerOffset = GetRelativeCalcerOffset(embeddingFeatureIdx, calcerId) * docCount;
            const size_t calculatedFeaturesSize = docCount * calcer->FeatureCount();

            auto currentResult = TArrayRef<float>(
                result.data() + calcerOffset,
                result.data() + calcerOffset + calculatedFeaturesSize
            );
            for (ui32 docId: xrange(docCount)) {
                calcer->Compute(
                    embeddingFeature[docId],
                    TOutputFloatIterator(currentResult.data() + docId, docCount, currentResult.size())
                );
            }
        }
    }

    void TEmbeddingProcessingCollection::SaveHeader(IOutputStream* stream) const {
        using namespace flatbuffers;
        FlatBufferBuilder builder(16);

        auto calcerId = FBSerializeGuidArray(builder, FeatureCalcerId);
        auto perEmbeddingFeatureCalcers = FBSerializeAdjacencyList(builder, PerEmbeddingFeatureCalcers);
        auto header = NCatBoostFbs::NEmbeddings::CreateTCollectionHeader(builder, calcerId, perEmbeddingFeatureCalcers);
        builder.Finish(header);
        ::Save(stream, static_cast<ui64>(builder.GetSize()));
        stream->Write(builder.GetBufferPointer(), builder.GetSize());
    }

    void TEmbeddingProcessingCollection::LoadHeader(IInputStream* stream) {
        ui64 DataHeaderSize;
        ::Load(stream, DataHeaderSize);
        TArrayHolder<ui8> arrayHolder(new ui8[DataHeaderSize]);
        const ui32 loadedBytes = stream->Load(arrayHolder.Get(), DataHeaderSize);
        CB_ENSURE(
            loadedBytes == DataHeaderSize,
            "Failed to deserialize: Failed to load EmbeddingProcessingCollection header"
        );
        {
            flatbuffers::Verifier verifier(arrayHolder.Get(), loadedBytes, 64 /* max depth */, 256000000 /* max tables */);
            CB_ENSURE(NCatBoostFbs::NEmbeddings::VerifyTCollectionHeaderBuffer(verifier), "Flatbuffers model verification failed");
        }
        auto headerTable = flatbuffers::GetRoot<NCatBoostFbs::NEmbeddings::TCollectionHeader>(arrayHolder.Get());
        CB_ENSURE(headerTable->CalcerId(), "There should be at least one calcer in TEmbeddingProcessingCollection");
        FBDeserializeGuidArray(*headerTable->CalcerId(), &FeatureCalcerId);
        CB_ENSURE(headerTable->PerEmbeddingFeatureCalcers(), "No PerEmbeddingFeatureCalcers found in TEmbeddingProcessingCollection");
        FBDeserializeAdjacencyList(*headerTable->PerEmbeddingFeatureCalcers(), &PerEmbeddingFeatureCalcers);
    }

    void TEmbeddingProcessingCollection::Save(IOutputStream* s) const {

        TCountingOutput stream(s);

        stream.Write(StringIdentifier.data(), IdentifierSize);
        AddPadding(&stream, SerializationAlignment);

        SaveHeader(&stream);

        for (ui32 calcerId : xrange(FeatureCalcers.size())) {
            flatbuffers::FlatBufferBuilder builder;

            const auto fbsPartGuid = CreateFbsGuid(FeatureCalcerId[calcerId]);
            auto collectionPart = NCatBoostFbs::NEmbeddings::CreateTCollectionPart(
                builder,
                NCatBoostFbs::NEmbeddings::EPartType::EPartType_EmbeddingCalcer,
                &fbsPartGuid);
            builder.Finish(collectionPart);

            ::Save(&stream, static_cast<ui64>(builder.GetSize()));
            stream.Write(builder.GetBufferPointer(), builder.GetSize());

            TEmbeddingCalcerSerializer::Save(&stream, *FeatureCalcers[calcerId]);
        }
    }

    void TEmbeddingProcessingCollection::DefaultInit(TCountingInput s) {
        std::array<char, IdentifierSize> stringIdentifier;
        const auto identifierSize = s.Load(stringIdentifier.data(), IdentifierSize);
        CB_ENSURE(
                IdentifierSize == identifierSize &&
                stringIdentifier == StringIdentifier,
                "Failed to deserialize: Couldn't load magic"
        );
        SkipPadding(&s, SerializationAlignment);
        LoadHeader(&s);

        FeatureCalcers.resize(FeatureCalcerId.size());
    }

    void TEmbeddingProcessingCollection::Load(IInputStream* stream) {
        DefaultInit(stream);
        THashMap<TGuid, ui32> guidId;
        for (ui32 i = 0; i < FeatureCalcerId.size(); i++) {
            CB_ENSURE_INTERNAL(!guidId.contains(FeatureCalcerId[i]), "Failed to deserialize: Get duplicated guid");
            guidId[FeatureCalcerId[i]] = i;
        }

        for (ui32 i = 0; i < FeatureCalcerId.size(); i++) {
            ui64 headerSize;
            ::Load(stream, headerSize);

            TArrayHolder<ui8> buffer(new ui8[headerSize]);
            const ui32 loadedBytes = stream->Load(buffer.Get(), headerSize);
            CB_ENSURE(
                loadedBytes == headerSize,
                "Failed to deserialize: Failed to load collection part"
            );

            auto collectionPart = flatbuffers::GetRoot<NCatBoostFbs::NEmbeddings::TCollectionPart>(buffer.Get());
            const auto partId = GuidFromFbs(collectionPart->Id());

            if (collectionPart->PartType() == NCatBoostFbs::NEmbeddings::EPartType_EmbeddingCalcer) {
                TEmbeddingFeatureCalcerPtr calcer = TEmbeddingCalcerSerializer::Load(stream);
                FeatureCalcers[guidId[partId]] = calcer;
                CB_ENSURE(partId == calcer->Id(), "Failed to deserialize: CalcerId not equal to PartId");
            } else {
                CB_ENSURE(false, "Failed to deserialize: Unknown part type");
            }
        }
        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

    void TEmbeddingProcessingCollection::LoadNonOwning(TMemoryInput* in) {
        DefaultInit(in);
        THashMap<TGuid, ui32> guidId;
        for (ui32 i = 0; i < FeatureCalcerId.size(); i++) {
            CB_ENSURE_INTERNAL(!guidId.contains(FeatureCalcerId[i]), "Failed to deserialize: Get duplicated guid");
            guidId[FeatureCalcerId[i]] = i;
        }

        for (ui32 i = 0; i < FeatureCalcerId.size(); i++) {
            ui64 headerSize;
            ::Load(in, headerSize);

            CB_ENSURE(
                    in->Avail() >= headerSize,
                    "Failed to deserialize: Failed to load collection part"
            );

            auto collectionPart = flatbuffers::GetRoot<NCatBoostFbs::NEmbeddings::TCollectionPart>(in->Buf());
            in->Skip(headerSize);

            const auto partId = GuidFromFbs(collectionPart->Id());

            if (collectionPart->PartType() == NCatBoostFbs::NEmbeddings::EPartType_EmbeddingCalcer) {
                TEmbeddingFeatureCalcerPtr calcer = TEmbeddingCalcerSerializer::Load(in);
                FeatureCalcers[guidId[partId]] = calcer;
                CB_ENSURE(partId == calcer->Id(), "Failed to deserialize: CalcerId not equal to PartId");
            } else {
                CB_ENSURE(false, "Failed to deserialize: Unknown part type");
            }
        }

        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

}
