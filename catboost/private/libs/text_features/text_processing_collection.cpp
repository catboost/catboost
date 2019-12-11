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
        for (ui32 docId: xrange(docCount)) {
            Tokenizer->Tokenize(textFeature[docId], &tokens[docId]);
        }

        for (ui32 dictionaryId: PerFeatureDictionaries[textFeatureIdx]) {
            const auto& dictionary = Dictionaries[dictionaryId];
            const ui32 tokenizedFeatureIdx = GetTokenizedFeatureId(textFeatureIdx, dictionaryId);

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

        auto calcerId = FBSerializeGuidArray(builder, FeatureCalcerId);
        auto dictionaryId = FBSerializeGuidArray(builder, DictionaryId);
        auto perFeatureDictionaries = FBSerializeAdjacencyList(builder, PerFeatureDictionaries);
        auto perTokenizedFeatureCalcers = FBSerializeAdjacencyList(builder, PerTokenizedFeatureCalcers);

        NCatBoostFbs::TCollectionHeaderBuilder headerBuilder(builder);
        headerBuilder.add_CalcerId(calcerId);
        headerBuilder.add_DictionaryId(dictionaryId);
        headerBuilder.add_PerFeatureDictionaries(perFeatureDictionaries);
        headerBuilder.add_PerTokenizedFeatureCalcers(perTokenizedFeatureCalcers);
        auto header = headerBuilder.Finish();
        builder.Finish(header);

        ::Save(stream, static_cast<ui64>(builder.GetSize()));
        stream->Write(builder.GetBufferPointer(), builder.GetSize());
    }

    void TTextProcessingCollection::LoadHeader(IInputStream* stream) {
        ui64 textDataHeaderSize;
        ::Load(stream, textDataHeaderSize);
        TArrayHolder<ui8> arrayHolder = new ui8[textDataHeaderSize];
        const ui32 loadedBytes = stream->Load(arrayHolder.Get(), textDataHeaderSize);
        CB_ENSURE(
            loadedBytes == textDataHeaderSize,
            "Failed to deserialize: Failed to load TextProcessingCollection header"
        );

        auto headerTable = flatbuffers::GetRoot<NCatBoostFbs::TCollectionHeader>(arrayHolder.Get());
        FBDeserializeGuidArray(*headerTable->CalcerId(), &FeatureCalcerId);
        FBDeserializeGuidArray(*headerTable->DictionaryId(), &DictionaryId);
        FBDeserializeAdjacencyList(*headerTable->PerFeatureDictionaries(), &PerFeatureDictionaries);
        FBDeserializeAdjacencyList(*headerTable->PerTokenizedFeatureCalcers(), &PerTokenizedFeatureCalcers);
    }

    void TTextProcessingCollection::Save(IOutputStream* s) const {
        TCountingOutput stream(s);

        stream.Write(StringIdentifier.data(), IdentifierSize);
        AddPadding(&stream, SerializationAlignment);

        SaveHeader(&stream);

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

        for (ui32 dictionaryId : xrange(Dictionaries.size())) {
            flatbuffers::FlatBufferBuilder builder;

            const auto fbsPartGuid = CreateFbsGuid(DictionaryId[dictionaryId]);
            auto collectionPart = NCatBoostFbs::CreateTCollectionPart(
                builder,
                NCatBoostFbs::EPartType::EPartType_Dictionary,
                &fbsPartGuid);
            builder.Finish(collectionPart);

            ::Save(&stream, static_cast<ui64>(builder.GetSize()));
            stream.Write(builder.GetBufferPointer(), builder.GetSize());

            Dictionaries[dictionaryId]->Save(&stream);
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

    void TTextProcessingCollection::Load(IInputStream* s) {
        TCountingInput stream(s);

        std::array<char, IdentifierSize> stringIdentifier;
        const auto identifierSize = stream.Load(stringIdentifier.data(), IdentifierSize);
        CB_ENSURE(
            IdentifierSize == identifierSize &&
                stringIdentifier == StringIdentifier,
            "Failed to deserialize: Couldn't load magic"
        );
        SkipPadding(&stream, SerializationAlignment);

        LoadHeader(&stream);

        THashMap<TGuid, ui32> guidId;
        for (ui32 i = 0; i < FeatureCalcerId.size(); i++) {
            CB_ENSURE_INTERNAL(!guidId.contains(FeatureCalcerId[i]), "Failed to deserialize: Get duplicated guid");
            guidId[FeatureCalcerId[i]] = i;
        }

        for (ui32 i = 0; i < DictionaryId.size(); i++) {
            CB_ENSURE_INTERNAL(!guidId.contains(DictionaryId[i]), "Failed to deserialize: Get duplicated guid");
            guidId[DictionaryId[i]] = i;
        }

        FeatureCalcers.resize(FeatureCalcerId.size());
        Dictionaries.resize(DictionaryId.size());

        ui64 headerSize;
        while (TryLoad(&stream, headerSize)) {
            TArrayHolder<ui8> buffer = new ui8[headerSize];
            const ui32 loadedBytes = stream.Load(buffer.Get(), headerSize);
            CB_ENSURE(
                loadedBytes == headerSize,
                "Failed to deserialize: Failed to load collection part"
            );

            auto collectionPart = flatbuffers::GetRoot<NCatBoostFbs::TCollectionPart>(buffer.Get());
            const auto partId = GuidFromFbs(collectionPart->Id());

            if (collectionPart->PartType() == NCatBoostFbs::EPartType_FeatureCalcer) {
                TTextFeatureCalcerPtr calcer = TTextCalcerSerializer::Load(&stream);
                FeatureCalcers[guidId[partId]] = calcer;
                CB_ENSURE(partId == calcer->Id(), "Failed to deserialize: CalcerId not equal to PartId");
            } else if (collectionPart->PartType() == NCatBoostFbs::EPartType_Dictionary) {
                auto dictionary = MakeIntrusive<TDictionaryProxy>();
                dictionary->Load(&stream);
                CB_ENSURE(
                    partId == dictionary->Id(),
                    "Failed to deserialize: DictionaryId not equal to PartId"
                );

                Dictionaries[guidId[partId]] = std::move(dictionary);
            } else {
                CB_ENSURE(false, "Failed to deserialize: Unknown part type");
            }
        }

        CB_ENSURE(
            AllOf(FeatureCalcers, [](const TTextFeatureCalcerPtr& calcerPtr) {
                return calcerPtr;
            } ),
            "Failed to deserialize: Some of calcers are missing"
        );

        CB_ENSURE(
            AllOf(Dictionaries, [](const TDictionaryPtr& dictionaryPtr) {
                return dictionaryPtr;
            } ),
            "Failed to deserialize: Some of dictionaries are missing"
        );

        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

    TTextProcessingCollection::TTextProcessingCollection(
        TVector<TTextFeatureCalcerPtr> calcers,
        TVector<TDictionaryPtr> dictionaries,
        TVector<TVector<ui32>> perFeatureDictionaries,
        TVector<TVector<ui32>> perTokenizedFeatureCalcers,
        TTokenizerPtr tokenizer)
        : Tokenizer(std::move(tokenizer))
        , Dictionaries(std::move(dictionaries))
        , FeatureCalcers(std::move(calcers))
        , PerFeatureDictionaries(std::move(perFeatureDictionaries))
        , PerTokenizedFeatureCalcers(std::move(perTokenizedFeatureCalcers)) {

        FeatureCalcerId.resize(FeatureCalcers.size());
        for (ui32 idx : xrange(FeatureCalcers.size())) {
            FeatureCalcerId[idx] = FeatureCalcers[idx]->Id();
        }

        DictionaryId.resize(Dictionaries.size());
        for (ui32 idx : xrange(Dictionaries.size())) {
            DictionaryId[idx] = Dictionaries[idx]->Id();
        }

        CalcRuntimeData();
        CheckPerFeatureIdx();
    }

    ui32 TTextProcessingCollection::GetTokenizedFeatureId(ui32 textFeatureIdx, ui32 dictionaryIdx) const {
        return TokenizedFeatureId.at(std::make_pair(textFeatureIdx, dictionaryIdx));
    }

    void TTextProcessingCollection::CalcRuntimeData() {
        ui32 tokenizedFeatureIdx = 0;
        ui32 currentOffset = 0;
        for (ui32 textFeatureIdx: xrange(PerFeatureDictionaries.size())) {
            for (ui32 dictionaryId: PerFeatureDictionaries[textFeatureIdx]) {
                auto pairIdx = std::make_pair(textFeatureIdx, dictionaryId);
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
        const ui32 firstDictionary = PerFeatureDictionaries[textFeatureIdx][0];
        const ui32 tokenizedFeature = GetTokenizedFeatureId(textFeatureIdx, firstDictionary);
        return PerTokenizedFeatureCalcers[tokenizedFeature][0];
    }

    ui32 TTextProcessingCollection::GetAbsoluteCalcerOffset(ui32 calcerIdx) const {
        return FeatureCalcerOffset.at(calcerIdx);
    }

    ui32 TTextProcessingCollection::GetRelativeCalcerOffset(ui32 textFeatureIdx, ui32 calcerIdx) const {
        return GetAbsoluteCalcerOffset(calcerIdx) - GetAbsoluteCalcerOffset(GetFirstTextFeatureCalcer(textFeatureIdx));
    }

    void TTextProcessingCollection::CheckPerFeatureIdx() const {
        for (ui32 featureId: xrange(PerFeatureDictionaries.size())) {
            for (ui32 dictionaryId: PerFeatureDictionaries[featureId]) {
                CB_ENSURE(
                    dictionaryId < Dictionaries.size(),
                    "For feature id=" << featureId << " specified dictionary id=" << dictionaryId
                        << " which is greater than number of dictionaries"
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
            DictionaryId,
            FeatureCalcerId,
            PerFeatureDictionaries,
            PerTokenizedFeatureCalcers,
            TokenizedFeatureId
        ) == std::tie(
            rhs.DictionaryId,
            rhs.FeatureCalcerId,
            rhs.PerFeatureDictionaries,
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

        for (const auto& dictionaryId: PerFeatureDictionaries[textFeatureId]) {
            const ui32 tokenizedFeatureId = GetTokenizedFeatureId(textFeatureId, dictionaryId);
            for (ui32 calcerId: PerTokenizedFeatureCalcers[tokenizedFeatureId]) {
                sum += FeatureCalcers[calcerId]->FeatureCount();
            }
        }

        return sum;
    }

    ui32 TTextProcessingCollection::GetTextFeatureCount() const {
        return PerFeatureDictionaries.size();
    }

    ui32 TTextProcessingCollection::GetTokenizedFeatureCount() const {
        return PerTokenizedFeatureCalcers.size();
    }

    ui32 TTextProcessingCollection::GetAbsoluteCalcerOffset(const TGuid& calcerGuid) const {
        CB_ENSURE(
            CalcerGuidToFlatIdx.contains(calcerGuid),
            "There is no calcer with " << LabeledOutput(calcerGuid)
        );
        return GetAbsoluteCalcerOffset(CalcerGuidToFlatIdx.at(calcerGuid));
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
        for (ui32 textFeatureId : xrange(PerFeatureDictionaries.size())) {
            for (ui32 dictionaryId : PerFeatureDictionaries[textFeatureId]) {
                const ui32 tokenizedFeatureId = GetTokenizedFeatureId(textFeatureId, dictionaryId);
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
