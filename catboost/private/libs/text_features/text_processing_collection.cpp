#include "text_processing_collection.h"
#include "helpers.h"

#include <catboost/libs/helpers/serialization.h>
#include <catboost/private/libs/text_features/flatbuffers/text_processing_collection.fbs.h>
#include <catboost/private/libs/text_processing/text_column_builder.h>

#include <util/stream/length.h>
#include <util/system/byteorder.h>
#include <util/generic/ylimits.h>

#include <cstring>

namespace NCB {
    static void CalcFeatures(
        const TVector<TTokensWithBuffer>& tokens,
        const TDictionaryProxy& dictionary,
        const TTextFeatureCalcer& calcer,
        TArrayRef<float> result
    ) {
        const ui64 docCount = tokens.size();
        for (ui32 docId: xrange(docCount)) {
            TText text = dictionary.Apply(tokens[docId].View);

            calcer.Compute(
                text,
                TOutputFloatIterator(result.data() + docId, docCount, result.size())
            );
        }
    }

    static void TokenizeTextFeature(
        TConstArrayRef<TStringBuf> textFeature,
        size_t docCount,
        TTokenizerPtr tokenizer,
        TVector<TTokensWithBuffer>* tokens
    ) {
        for (ui32 docId: xrange(docCount)) {
            tokenizer->Tokenize(textFeature[docId], &(*tokens)[docId]);
        }
    }

    void TTextProcessingCollection::CalcFeatures(
        TConstArrayRef<TStringBuf> textFeature,
        ui32 textFeatureIdx,
        size_t docCount,
        TArrayRef<float> result
    ) const {
        CB_ENSURE(
            result.size() >= NumberOfOutputFeatures(textFeatureIdx) * docCount,
            "Proposed result buffer has size less than text processing produce"
        );

        TVector<TTokensWithBuffer> tokens;
        tokens.yresize(docCount);
        TTokenizerPtr previousTokenizer;

        for (ui32 digitizerId: PerFeatureDigitizers[textFeatureIdx]) {
            const auto& dictionary = Digitizers[digitizerId].Dictionary;
            const ui32 tokenizedFeatureIdx = GetTokenizedFeatureId(textFeatureIdx, digitizerId);

            if (!previousTokenizer || Digitizers[digitizerId].Tokenizer != previousTokenizer) {
                TokenizeTextFeature(textFeature, docCount, Digitizers[digitizerId].Tokenizer, &tokens);
                previousTokenizer = Digitizers[digitizerId].Tokenizer;
            }

            for (ui32 calcerId: PerTokenizedFeatureCalcers[tokenizedFeatureIdx]) {
                const auto& calcer = FeatureCalcers[calcerId];

                const size_t calcerOffset = GetRelativeCalcerOffset(textFeatureIdx, calcerId) * docCount;
                const size_t calculatedFeaturesSize = docCount * calcer->FeatureCount();

                auto currentResult = TArrayRef<float>(
                    result.data() + calcerOffset,
                    result.data() + calcerOffset + calculatedFeaturesSize
                );
                NCB::CalcFeatures(tokens, *dictionary, *calcer, currentResult);
            }
        }
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
        using namespace NCatBoostFbs;

        TVector<Offset<AdjacencyList>> fbBipartiteGraph;
        fbBipartiteGraph.reserve(bipartiteGraph.size());

        for (const auto& adjacencyList: bipartiteGraph) {
            fbBipartiteGraph.push_back(CreateAdjacencyListDirect(builder, &adjacencyList));
        }

        return builder.CreateVector(fbBipartiteGraph.data(), fbBipartiteGraph.size());
    }

    static void FBDeserializeAdjacencyList(
        const flatbuffers::Vector<flatbuffers::Offset<NCatBoostFbs::AdjacencyList>>& fbBipartiteGraph,
        TVector<TVector<ui32>>* bipartiteGraph
    ) {
        bipartiteGraph->clear();
        bipartiteGraph->reserve(fbBipartiteGraph.size());

        for (auto fbAdjacencyList : fbBipartiteGraph) {
            const flatbuffers::Vector<uint32_t>* ids = fbAdjacencyList->Ids();
            bipartiteGraph->emplace_back(ids->begin(), ids->end());
        }
    }

    void TTextProcessingCollection::SaveHeader(IOutputStream* stream) const {
        using namespace flatbuffers;
        FlatBufferBuilder builder;

        auto tokenizerId = FBSerializeGuidArray(builder, TokenizerId);
        auto dictionaryId = FBSerializeGuidArray(builder, DictionaryId);
        auto calcerId = FBSerializeGuidArray(builder, FeatureCalcerId);
        auto perFeatureDigitizers = FBSerializeAdjacencyList(builder, PerFeatureDigitizers);
        auto perTokenizedFeatureCalcers = FBSerializeAdjacencyList(builder, PerTokenizedFeatureCalcers);

        NCatBoostFbs::TCollectionHeaderBuilder headerBuilder(builder);
        headerBuilder.add_TokenizerId(tokenizerId);
        headerBuilder.add_DictionaryId(dictionaryId);
        headerBuilder.add_CalcerId(calcerId);
        headerBuilder.add_PerFeatureDigitizers(perFeatureDigitizers);
        headerBuilder.add_PerTokenizedFeatureCalcers(perTokenizedFeatureCalcers);
        auto header = headerBuilder.Finish();
        builder.Finish(header);

        ::Save(stream, static_cast<ui64>(builder.GetSize()));
        stream->Write(builder.GetBufferPointer(), builder.GetSize());
    }

    void TTextProcessingCollection::LoadHeader(IInputStream* stream) {
        ui64 textDataHeaderSize;
        ::Load(stream, textDataHeaderSize);
        TArrayHolder<ui8> arrayHolder(new ui8[textDataHeaderSize]);
        const ui32 loadedBytes = stream->Load(arrayHolder.Get(), textDataHeaderSize);
        CB_ENSURE(
            loadedBytes == textDataHeaderSize,
            "Failed to deserialize: Failed to load TextProcessingCollection header"
        );
        auto headerTable = flatbuffers::GetRoot<NCatBoostFbs::TCollectionHeader>(arrayHolder.Get());
        FBDeserializeGuidArray(*headerTable->TokenizerId(), &TokenizerId);
        FBDeserializeGuidArray(*headerTable->DictionaryId(), &DictionaryId);
        FBDeserializeGuidArray(*headerTable->CalcerId(), &FeatureCalcerId);
        FBDeserializeAdjacencyList(*headerTable->PerFeatureDigitizers(), &PerFeatureDigitizers);
        FBDeserializeAdjacencyList(*headerTable->PerTokenizedFeatureCalcers(), &PerTokenizedFeatureCalcers);
    }

    static void SaveGuidAndType(
        const TGuid& guid,
        NCatBoostFbs::EPartType type,
        TCountingOutput* stream
    ) {
        flatbuffers::FlatBufferBuilder builder;
        const auto fbsPartGuid = CreateFbsGuid(guid);
        auto collectionPart = NCatBoostFbs::CreateTCollectionPart(builder, type, &fbsPartGuid);
        builder.Finish(collectionPart);

        ::Save(stream, static_cast<ui64>(builder.GetSize()));
        stream->Write(builder.GetBufferPointer(), builder.GetSize());
    }

    void TTextProcessingCollection::Save(IOutputStream* s) const {
        TCountingOutput stream(s);

        stream.Write(StringIdentifier.data(), IdentifierSize);
        AddPadding(&stream, SerializationAlignment);

        SaveHeader(&stream);

        for (ui32 digitizerId : xrange(Digitizers.size())) {
            SaveGuidAndType(TokenizerId[digitizerId], NCatBoostFbs::EPartType::EPartType_Tokenizer, &stream);
            Digitizers[digitizerId].Tokenizer->Save(&stream);
            SaveGuidAndType(DictionaryId[digitizerId], NCatBoostFbs::EPartType::EPartType_Dictionary, &stream);
            Digitizers[digitizerId].Dictionary->Save(&stream);
        }

        for (ui32 calcerId : xrange(FeatureCalcers.size())) {
            flatbuffers::FlatBufferBuilder builder;

            const auto fbsPartGuid = CreateFbsGuid(FeatureCalcerId[calcerId]);
            auto collectionPart = NCatBoostFbs::CreateTCollectionPart(
                builder,
                NCatBoostFbs::EPartType::EPartType_FeatureCalcer,
                &fbsPartGuid);
            builder.Finish(collectionPart);

            ::Save(&stream, static_cast<ui64>(builder.GetSize()));
            stream.Write(builder.GetBufferPointer(), builder.GetSize());

            TTextCalcerSerializer::Save(&stream, *FeatureCalcers[calcerId]);
        }
        {
            auto guid = TGuid();
            SaveGuidAndType(guid, NCatBoostFbs::EPartType::EPartType_Terminate, &stream);
        }
    }

    template <class T>
    static bool TryLoad(IInputStream* stream, T& value) {
        const ui32 readLen = stream->Load(&value, sizeof(T));
        CB_ENSURE_INTERNAL(
            readLen == 0 || readLen == sizeof(T),
            "Failed to deserialize: only half of header was read"
        );
        return readLen == sizeof(T);
    }

    THashMap<TGuid, ui32> TTextProcessingCollection::CreateComponentGuidsMapping() const {
        THashMap<TGuid, ui32> guidId;
        for (ui32 i = 0; i < TokenizerId.size(); i++) {
            CB_ENSURE_INTERNAL(!guidId.contains(TokenizerId[i]), "Failed to deserialize: Get duplicated guid");
            guidId[TokenizerId[i]] = i;
        }

        for (ui32 i = 0; i < DictionaryId.size(); i++) {
            CB_ENSURE_INTERNAL(!guidId.contains(DictionaryId[i]), "Failed to deserialize: Get duplicated guid");
            guidId[DictionaryId[i]] = i;
        }

        for (ui32 i = 0; i < FeatureCalcerId.size(); i++) {
            CB_ENSURE_INTERNAL(!guidId.contains(FeatureCalcerId[i]), "Failed to deserialize: Get duplicated guid");
            guidId[FeatureCalcerId[i]] = i;
        }
        return guidId;
    }

    void TTextProcessingCollection::CheckForMissingParts() const {
        CB_ENSURE(
                AllOf(Digitizers, [](const TDigitizer& digitizer) {
                    return digitizer.Tokenizer && digitizer.Dictionary;
                }),
                "Failed to deserialize: Some of tokenizers or dictionaries are missing"
        );

        CB_ENSURE(
                AllOf(FeatureCalcers, [](const TTextFeatureCalcerPtr& calcerPtr) {
                    return calcerPtr;
                }),
                "Failed to deserialize: Some of calcers are missing"
        );
    }

    void TTextProcessingCollection::DefaultInit(TCountingInput s) {
        std::array<char, IdentifierSize> stringIdentifier;
        const auto identifierSize = s.Load(stringIdentifier.data(), IdentifierSize);
        CB_ENSURE(
                IdentifierSize == identifierSize &&
                stringIdentifier == StringIdentifier,
                "Failed to deserialize: Couldn't load magic"
        );
        SkipPadding(&s, SerializationAlignment);
        LoadHeader(&s);

        CB_ENSURE(TokenizerId.size() == DictionaryId.size(), "Failed to deserialize: TokenizerId.size should be equal to DictionaryId.size");
        Digitizers.resize(TokenizerId.size());
        FeatureCalcers.resize(FeatureCalcerId.size());
    }

    void TTextProcessingCollection::Load(IInputStream* stream) {
        DefaultInit(stream);
        auto guidId = CreateComponentGuidsMapping();

        ui64 headerSize;
        while (TryLoad(stream, headerSize)) {
            TArrayHolder<ui8> buffer(new ui8[headerSize]);
            const ui32 loadedBytes = stream->Load(buffer.Get(), headerSize);
            CB_ENSURE(
                loadedBytes == headerSize,
                "Failed to deserialize: Failed to load collection part"
            );

            auto collectionPart = flatbuffers::GetRoot<NCatBoostFbs::TCollectionPart>(buffer.Get());
            const auto partId = GuidFromFbs(collectionPart->Id());

            if (collectionPart->PartType() == NCatBoostFbs::EPartType_Tokenizer) {
                auto tokenizer = MakeIntrusive<TTokenizer>();
                tokenizer->Load(stream);
                CB_ENSURE(
                    partId == tokenizer->Id(),
                    "Failed to deserialize: TokenizerId not equal to PartId"
                );

                Digitizers[guidId[partId]].Tokenizer = std::move(tokenizer);
            } else if (collectionPart->PartType() == NCatBoostFbs::EPartType_Dictionary) {
                auto dictionary = MakeIntrusive<TDictionaryProxy>();
                dictionary->Load(stream);
                CB_ENSURE(
                    partId == dictionary->Id(),
                    "Failed to deserialize: DictionaryId not equal to PartId"
                );

                Digitizers[guidId[partId]].Dictionary = std::move(dictionary);
            } else if (collectionPart->PartType() == NCatBoostFbs::EPartType_FeatureCalcer) {
                TTextFeatureCalcerPtr calcer = TTextCalcerSerializer::Load(stream);
                FeatureCalcers[guidId[partId]] = calcer;
                CB_ENSURE(partId == calcer->Id(), "Failed to deserialize: CalcerId not equal to PartId");
            } else if (collectionPart->PartType() == NCatBoostFbs::EPartType_Terminate){
                break;
            } else {
                CB_ENSURE(false, "Failed to deserialize: Unknown part type");
            }
        }

        CheckForMissingParts();
        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

    void TTextProcessingCollection::LoadNonOwning(TMemoryInput* in) {
        DefaultInit(in);
        auto guidId = CreateComponentGuidsMapping();

        ui64 headerSize;
        while (TryLoad(in, headerSize)) {
            CB_ENSURE(
                    in->Avail() >= headerSize,
                    "Failed to deserialize: Failed to load collection part"
            );

            auto collectionPart = flatbuffers::GetRoot<NCatBoostFbs::TCollectionPart>(in->Buf());
            in->Skip(headerSize);

            const auto partId = GuidFromFbs(collectionPart->Id());

            if (collectionPart->PartType() == NCatBoostFbs::EPartType_Tokenizer) {
                auto tokenizer = MakeIntrusive<TTokenizer>();
                tokenizer->Load(in);
                CB_ENSURE(
                        partId == tokenizer->Id(),
                        "Failed to deserialize: TokenizerId not equal to PartId"
                );

                Digitizers[guidId[partId]].Tokenizer = std::move(tokenizer);
            } else if (collectionPart->PartType() == NCatBoostFbs::EPartType_Dictionary) {
                auto dictionary = MakeIntrusive<TDictionaryProxy>();
                dictionary->LoadNonOwning(in);
                CB_ENSURE(
                        partId == dictionary->Id(),
                        "Failed to deserialize: DictionaryId not equal to PartId"
                );

                Digitizers[guidId[partId]].Dictionary = std::move(dictionary);
            } else if (collectionPart->PartType() == NCatBoostFbs::EPartType_FeatureCalcer) {
                TTextFeatureCalcerPtr calcer = TTextCalcerSerializer::Load(in);
                FeatureCalcers[guidId[partId]] = calcer;
                CB_ENSURE(partId == calcer->Id(), "Failed to deserialize: CalcerId not equal to PartId");
            } else if (collectionPart->PartType() == NCatBoostFbs::EPartType_Terminate){
                break;
            } else {
                CB_ENSURE(false, "Failed to deserialize: Unknown part type");
            }
        }

        CheckForMissingParts();
        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

    TTextProcessingCollection::TTextProcessingCollection(
        TVector<TDigitizer> digitizers,
        TVector<TTextFeatureCalcerPtr> calcers,
        TVector<TVector<ui32>> perFeatureDigitizers,
        TVector<TVector<ui32>> perTokenizedFeatureCalcers
    )
        : Digitizers(std::move(digitizers))
        , FeatureCalcers(std::move(calcers))
        , PerFeatureDigitizers(std::move(perFeatureDigitizers))
        , PerTokenizedFeatureCalcers(std::move(perTokenizedFeatureCalcers))
    {
        TokenizerId.resize(Digitizers.size());
        DictionaryId.resize(Digitizers.size());
        for (ui32 idx : xrange(Digitizers.size())) {
            TokenizerId[idx] = Digitizers[idx].Tokenizer->Id();
            DictionaryId[idx] = Digitizers[idx].Dictionary->Id();
        }

        FeatureCalcerId.resize(FeatureCalcers.size());
        for (ui32 idx : xrange(FeatureCalcers.size())) {
            FeatureCalcerId[idx] = FeatureCalcers[idx]->Id();
        }

        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

    ui32 TTextProcessingCollection::GetTokenizedFeatureId(ui32 textFeatureIdx, ui32 digitizerIdx) const {
        return TokenizedFeatureId.at(std::make_pair(textFeatureIdx, digitizerIdx));
    }

    void TTextProcessingCollection::CalcRuntimeData() {
        ui32 tokenizedFeatureIdx = 0;
        ui32 currentOffset = 0;
        for (ui32 textFeatureIdx: xrange(PerFeatureDigitizers.size())) {
            for (ui32 digitizerId: PerFeatureDigitizers[textFeatureIdx]) {
                auto pairIdx = std::make_pair(textFeatureIdx, digitizerId);
                TokenizedFeatureId[pairIdx] = tokenizedFeatureIdx;

                for (ui32 calcerId: PerTokenizedFeatureCalcers[tokenizedFeatureIdx]) {
                    FeatureCalcerOffset[calcerId] = currentOffset;
                    currentOffset += FeatureCalcers[calcerId]->FeatureCount();
                }

                tokenizedFeatureIdx++;
            }
        }

        for (ui32 calcerFlatIdx: xrange(FeatureCalcerId.size())) {
            CalcerGuidToFlatIdx[FeatureCalcerId[calcerFlatIdx]] = calcerFlatIdx;
        }
    }

    ui32 TTextProcessingCollection::GetFirstTextFeatureCalcer(ui32 textFeatureIdx) const {
        const ui32 firstDigitizer = PerFeatureDigitizers[textFeatureIdx][0];
        const ui32 tokenizedFeature = GetTokenizedFeatureId(textFeatureIdx, firstDigitizer);
        return PerTokenizedFeatureCalcers[tokenizedFeature][0];
    }

    ui32 TTextProcessingCollection::GetAbsoluteCalcerOffset(ui32 calcerIdx) const {
        return FeatureCalcerOffset.at(calcerIdx);
    }

    ui32 TTextProcessingCollection::GetRelativeCalcerOffset(ui32 textFeatureIdx, ui32 calcerIdx) const {
        return GetAbsoluteCalcerOffset(calcerIdx) - GetAbsoluteCalcerOffset(GetFirstTextFeatureCalcer(textFeatureIdx));
    }

    void TTextProcessingCollection::CheckPerFeatureIdx() const {
        for (ui32 featureId: xrange(PerFeatureDigitizers.size())) {
            for (ui32 digitizerId: PerFeatureDigitizers[featureId]) {
                CB_ENSURE(
                    digitizerId < Digitizers.size(),
                    "For feature id=" << featureId << " specified digitizer id=" << digitizerId
                        << " which is greater than number of digitizers"
                );
            }
        }

        for (ui32 tokenizedFeatureId: xrange(PerTokenizedFeatureCalcers.size())) {
            for (ui32 calcerId: PerTokenizedFeatureCalcers[tokenizedFeatureId]) {
                CB_ENSURE(
                    calcerId < FeatureCalcers.size(),
                    "For tokenized feature id=" << tokenizedFeatureId << " specified feature calcer id="
                        << calcerId << " which is greater than number of calcers"
                );
            }
        }
    }

    bool TTextProcessingCollection::operator==(const TTextProcessingCollection& rhs) {
        return std::tie(
            TokenizerId,
            DictionaryId,
            FeatureCalcerId,
            PerFeatureDigitizers,
            PerTokenizedFeatureCalcers,
            TokenizedFeatureId
        ) == std::tie(
            rhs.TokenizerId,
            rhs.DictionaryId,
            rhs.FeatureCalcerId,
            rhs.PerFeatureDigitizers,
            rhs.PerTokenizedFeatureCalcers,
            rhs.TokenizedFeatureId
        );
    }

    bool TTextProcessingCollection::operator!=(const TTextProcessingCollection& rhs) {
        return !(*this == rhs);
    }

    ui32 TTextProcessingCollection::TotalNumberOfOutputFeatures() const {
        ui32 sum = 0;
        for (const auto& tokenizedFeatureCalcers: PerTokenizedFeatureCalcers) {
            for (ui32 calcerId: tokenizedFeatureCalcers) {
                sum += FeatureCalcers[calcerId]->FeatureCount();
            }
        }
        return sum;
    }

    ui32 TTextProcessingCollection::NumberOfOutputFeatures(ui32 textFeatureId) const {
        ui32 sum = 0;

        for (const auto& digitizerId: PerFeatureDigitizers[textFeatureId]) {
            const ui32 tokenizedFeatureId = GetTokenizedFeatureId(textFeatureId, digitizerId);
            for (ui32 calcerId: PerTokenizedFeatureCalcers[tokenizedFeatureId]) {
                sum += FeatureCalcers[calcerId]->FeatureCount();
            }
        }

        return sum;
    }

    ui32 TTextProcessingCollection::GetTextFeatureCount() const {
        return PerFeatureDigitizers.size();
    }

    ui32 TTextProcessingCollection::GetTokenizedFeatureCount() const {
        return PerTokenizedFeatureCalcers.size();
    }

    ui32 TTextProcessingCollection::GetAbsoluteCalcerOffset(const TGuid& calcerGuid) const {
        auto it = CalcerGuidToFlatIdx.find(calcerGuid);
        CB_ENSURE(
            it != CalcerGuidToFlatIdx.end(),
            "There is no calcer with " << LabeledOutput(calcerGuid)
        );
        return GetAbsoluteCalcerOffset(it->second);
    }

    ui32 TTextProcessingCollection::GetRelativeCalcerOffset(ui32 textFeatureIdx, const TGuid& calcerGuid) const {
        return GetRelativeCalcerOffset(textFeatureIdx, CalcerGuidToFlatIdx.at(calcerGuid));
    }

    static void CreateEvaluatedCalcerFeatures(
        ui32 textFeatureIdx,
        const TTextFeatureCalcer& calcer,
        TVector<TEvaluatedFeature>* evaluatedFeatures
    ) {
        const TGuid& guid = calcer.Id();
        for (ui32 localId: xrange(calcer.FeatureCount())) {
            evaluatedFeatures->emplace_back(
                TEvaluatedFeature{
                    textFeatureIdx,
                    guid,
                    localId
                }
            );
        }
    }

    TVector<TEvaluatedFeature> TTextProcessingCollection::GetProducedFeatures() const {
        TVector<TEvaluatedFeature> evaluatedFeatures;
        for (ui32 textFeatureId : xrange(PerFeatureDigitizers.size())) {
            for (ui32 digitizerId : PerFeatureDigitizers[textFeatureId]) {
                const ui32 tokenizedFeatureId = GetTokenizedFeatureId(textFeatureId, digitizerId);
                for (ui32 calcerId: PerTokenizedFeatureCalcers[tokenizedFeatureId]) {
                    CreateEvaluatedCalcerFeatures(
                        textFeatureId,
                        *FeatureCalcers[calcerId],
                        &evaluatedFeatures
                    );
                }
            }
        }
        return evaluatedFeatures;
    }


} // NCB
