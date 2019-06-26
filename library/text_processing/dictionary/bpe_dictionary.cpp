#include "bpe_builder.h"
#include "serialization_helpers.h"

#include <util/digest/murmur.h>
#include <util/generic/array_ref.h>
#include <util/generic/hash_set.h>
#include <util/generic/maybe.h>
#include <util/generic/queue.h>
#include <util/string/split.h>

using namespace NTextProcessing::NDictionary;
using NTextProcessing::NDictionary::EUnknownTokenPolicy;
using TUnit = std::pair<TTokenId, TTokenId>;
using TTokenToUnit = std::pair<TTokenId, TUnit>;
using TUnitQueue = TPriorityQueue<TTokenToUnit, TVector<TTokenToUnit>, std::greater<TTokenToUnit>>;

static const char BPE_MAGIC[] = "MMapBpeDict";
static const size_t BPE_MAGIC_SIZE = Y_ARRAY_SIZE(BPE_MAGIC);  // yes, with terminating zero

template <typename TUnitToTokenId>
static void AddUnit(
    const TUnit& unit,
    const TUnitToTokenId& unitToTokenId,
    THashMap<TUnit, int>* unitCounts,
    TUnitQueue* minIds) {

    int countOfNewUnits = ++(*unitCounts)[unit];
    if (countOfNewUnits == 1) {
        auto maybeTokenId = unitToTokenId(unit);
        if (maybeTokenId) {
            minIds->emplace(*maybeTokenId, unit);
        }
    }
}

template <typename TUnitToTokenId>
static void UpdatePrevUnit(
    const TVector<TTokenId>& tokenIds,
    int idxStartInOld,
    int idxStartInNew,
    TTokenId changedToken,
    const TUnitToTokenId& unitToTokenId,
    THashMap<TUnit, int>* unitCounts,
    TUnitQueue* minIds) {
    if (idxStartInNew == 0) {
        return;
    }
    TUnit oldPrevUnit{tokenIds[idxStartInNew - 1], tokenIds[idxStartInOld]};
    (*unitCounts)[oldPrevUnit]--;

    TUnit newPrevUnit{tokenIds[idxStartInNew - 1], changedToken};
    AddUnit(newPrevUnit, unitToTokenId, unitCounts, minIds);
}

template <typename TUnitToTokenId>
static void UpdateNextUnit(
    const TVector<TTokenId>& tokenIds,
    int idxStartInOld,
    TTokenId changedToken,
    const TUnitToTokenId& unitToTokenId,
    THashMap<TUnit, int>* unitCounts,
    TUnitQueue* minIds) {

    if (idxStartInOld + 2 >= tokenIds.ysize()) {
        return;
    }
    TUnit oldNextUnit{tokenIds[idxStartInOld + 1], tokenIds[idxStartInOld + 2]};
    (*unitCounts)[oldNextUnit]--;

    TUnit newNextUnit{changedToken, tokenIds[idxStartInOld + 2]};
    AddUnit(newNextUnit, unitToTokenId, unitCounts, minIds);
}

// TODO(annaveronika): more efficient apply.
template <typename TStringVector, typename TUnitToTokenId>
static void ApplyImpl(
    TStringVector tokens,
    TVector<TTokenId>* tokenIds,
    const IDictionary* alphabet,
    const TUnitToTokenId& unitToTokenId,
    EUnknownTokenPolicy unknownTokenPolicy
) {
    tokenIds->clear();
    alphabet->Apply(tokens, tokenIds, unknownTokenPolicy);

    TPriorityQueue<TTokenToUnit, TVector<TTokenToUnit>, std::greater<TTokenToUnit>> minIds;
    THashMap<TUnit, int> unitCounts;

    for (size_t i = 0; i + 1 < tokenIds->size(); ++i) {
        auto unit = TUnit((*tokenIds)[i], (*tokenIds)[i + 1]);
        auto maybeTokenId = unitToTokenId(unit);
        if (!maybeTokenId) {
            continue;
        }
        auto unitId = *maybeTokenId;
        auto unitInCounts = unitCounts.find(unit);
        if (unitInCounts == unitCounts.end()) {
            minIds.push({unitId, unit});
            unitCounts[unit]++;
        } else {
            unitInCounts->second++;
        }
    }

    while (!minIds.empty()) {
        auto bestIdWithUnit = minIds.top();
        auto bestId = bestIdWithUnit.first;
        auto bestUnit = bestIdWithUnit.second;

        minIds.pop();
        if (unitCounts[bestUnit] == 0) {
            continue;
        }

        size_t i = 0, j = 0;
        while (i < tokenIds->size()) {
            if (i + 1 < tokenIds->size() && bestUnit.first == (*tokenIds)[i] && bestUnit.second == (*tokenIds)[i + 1]) {
                UpdatePrevUnit(*tokenIds, i, j, bestId, unitToTokenId, &unitCounts, &minIds);
                UpdateNextUnit(*tokenIds, i, bestId, unitToTokenId, &unitCounts, &minIds);

                (*tokenIds)[j] = bestId;
                unitCounts[bestUnit]--;
                i += 2;
            } else {
                (*tokenIds)[j] = (*tokenIds)[i];
                i += 1;
            }
            j += 1;
        }
        tokenIds->resize(j);
    }
}

TBpeDictionary::TBpeDictionary(TIntrusivePtr<TDictionary> alphabet)
    : Alphabet(alphabet)
{
}

TTokenId TBpeDictionary::Apply(TStringBuf) const {
    Y_ENSURE(false, "This method is unimplemented for TBpeDictionary.");
}

static std::function<TMaybe<TTokenId>(const TUnit& unit)> GetUnitToTokenIdFunc(
    const THashMap<std::pair<TTokenId, TTokenId>, TTokenId>& sourceTokenIdsToTokenId
) {
    return [&](const TUnit& unit) {
        auto it = sourceTokenIdsToTokenId.find(unit);
        return it == sourceTokenIdsToTokenId.end() ? Nothing() : TMaybe<TTokenId>(it->second);
    };
}

void TBpeDictionary::Apply(
    TConstArrayRef<TString> tokens,
    TVector<TTokenId>* tokensIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(
        tokens,
        tokensIds,
        Alphabet.Get(),
        GetUnitToTokenIdFunc(SourceTokenIdsToTokenId),
        unknownTokenPolicy
    );
}

void TBpeDictionary::Apply(
    TConstArrayRef<TStringBuf> tokens,
    TVector<TTokenId>* tokensIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(
        tokens,
        tokensIds,
        Alphabet.Get(),
        GetUnitToTokenIdFunc(SourceTokenIdsToTokenId),
        unknownTokenPolicy
    );
}

ui32 TBpeDictionary::Size() const {
    return Alphabet->Size() + BpeUnits.size();
}

TString TBpeDictionary::GetToken(TTokenId tokenId) const {
    TTokenId minId = GetMinTokenIdForUnits();
    if (tokenId < minId) {
        return Alphabet->GetToken(tokenId);
    }
    // TODO(nikitxskv): Add tokenId checks like in TDictionary.
    return StringTokens[tokenId - minId];
}

ui64 TBpeDictionary::GetCount(TTokenId tokenId) const {
    TTokenId minId = GetMinTokenIdForUnits();
    if (tokenId < minId) {
        return Alphabet->GetCount(tokenId);
    }
    // TODO(nikitxskv): Add tokenId checks like in TDictionary.
    return BpeUnits[tokenId - minId].Count;
}

TVector<TString> TBpeDictionary::GetTopTokens(ui32 /*topSize*/) const {
    Y_ENSURE(false, "This method is unimplemented for TBpeDictionary.");
}

void TBpeDictionary::ClearStatsData() {
    // TODO(nikitxksv): Implement this method.
}

TTokenId TBpeDictionary::GetUnknownTokenId() const {
    return Alphabet->GetUnknownTokenId();
}

TTokenId TBpeDictionary::GetEndOfSentenceTokenId() const {
    return Alphabet->GetEndOfSentenceTokenId();
}

TTokenId TBpeDictionary::GetMinUnusedTokenId() const {
    return Alphabet->GetMinUnusedTokenId() + BpeUnits.size();
}

void TBpeDictionary::SetAlphabet(TIntrusivePtr<TDictionary> alphabet) {
    Alphabet = alphabet;
}

TIntrusiveConstPtr<TDictionary> TBpeDictionary::GetAlphabet() const {
    return Alphabet.Get();
}

TString TBpeDictionary::GetBpeToken(TTokenId leftId, TTokenId rightId) const {
    if (Alphabet->GetDictionaryOptionsRef().TokenLevelType == ETokenLevelType::Word) {
        return TString::Join(GetToken(leftId), " ", GetToken(rightId));
    } else {
        Y_ASSERT(Alphabet->GetDictionaryOptionsRef().TokenLevelType == ETokenLevelType::Letter);
        return TString::Join(GetToken(leftId), GetToken(rightId));
    }
}

void TBpeDictionary::Save(IOutputStream* stream) const {
    for (const auto& unit : BpeUnits) {
        *stream << unit.Left << '\t' << unit.Right << '\t' << unit.Count << '\t' << GetBpeToken(unit.Left, unit.Right) << '\n';
    }
}

void TBpeDictionary::Load(IInputStream* stream) {
    TString line;
    while (stream->ReadLine(line)) {
        TBpeUnit unit;
        TString _;
        StringSplitter(line).Split('\t').Limit(4).CollectInto(&unit.Left, &unit.Right, &unit.Count, &_);
        BpeUnits.push_back(unit);
    }
    InitBpeTokens();
}

void TBpeDictionary::Load(const TString& dictionaryPath, const TString& bpePath) {
    TFileInput dictInput(dictionaryPath);
    Alphabet = MakeIntrusive<TDictionary>();
    Alphabet->Load(&dictInput);

    TFileInput bpeInput(bpePath);
    Load(&bpeInput);
}

void TBpeDictionary::Save(const TString& dictionaryPath, const TString& bpePath) const {
    TFileOutput dictionaryOutput(dictionaryPath);
    GetAlphabet()->Save(&dictionaryOutput);
    TFileOutput bpeDictionaryOutput(bpePath);
    Save(&bpeDictionaryOutput);
}

void TBpeDictionary::InitBpeTokens() {
    TTokenId curTokenId = GetMinTokenIdForUnits();
    for (const auto& unit : BpeUnits) {
        SourceTokenIdsToTokenId[std::pair<TTokenId, TTokenId>(unit.Left, unit.Right)] = curTokenId++;
        StringTokens.push_back(GetBpeToken(unit.Left, unit.Right));
    }
}

static ui64 MurmurHashFromUnit(const TUnit& unit, ui64 seed) {
    return MurmurHash<ui64>((void*)(&unit), sizeof(unit), seed);
}

TMMapBpeDictionary::TMMapBpeDictionary(TIntrusivePtr<TBpeDictionary> bpeDictionary)
    : Alphabet(MakeIntrusive<TMMapDictionary>(bpeDictionary->Alphabet))
    , BpeSize(bpeDictionary->SourceTokenIdsToTokenId.size())
{
    BuildBuckets(
        bpeDictionary->BpeUnits,
        [&](const TBpeDictionary::TBpeUnit& bpeUnit, ui64 seed) {
            const auto unit = std::make_pair(bpeUnit.Left, bpeUnit.Right);
            const auto hash = MurmurHashFromUnit(unit, seed);
            return std::make_pair(hash, bpeDictionary->SourceTokenIdsToTokenId.at(unit));
        },
        &SourceTokenIdsToTokenIdBuffer,
        &SourceTokenIdsToTokenIdSeed
    );
    SourceTokenIdsToTokenId = MakeArrayRef(SourceTokenIdsToTokenIdBuffer);
}

TMMapBpeDictionary::TMMapBpeDictionary(TIntrusivePtr<TMMapDictionary> alphabet)
    : Alphabet(alphabet)
{
}

TMMapBpeDictionary::TMMapBpeDictionary(
    TIntrusivePtr<TMMapDictionary> alphabet,
    const void* data,
    size_t size
)
    : Alphabet(alphabet)
{
    InitFromMemory(data, size);
}

TTokenId TMMapBpeDictionary::Apply(TStringBuf) const {
    Y_ENSURE(false, "This method is unimplemented for TMMapBpeDictionary.");
}

static std::function<TMaybe<TTokenId>(const TUnit& unit)> GetUnitToTokenIdFuncForMMap(
    TConstArrayRef<TBucket> sourceTokenIdsToTokenId,
    ui64 sourceTokenIdsToTokenIdSeed
) {
    return [=](const TUnit& unit) {
        const auto hash = MurmurHashFromUnit(unit, sourceTokenIdsToTokenIdSeed);
        const auto& bucket = sourceTokenIdsToTokenId[GetBucketIndex(hash, sourceTokenIdsToTokenId)];
        return bucket.Hash == hash ? TMaybe<TTokenId>(bucket.TokenId) : Nothing();
    };
}

void TMMapBpeDictionary::Apply(
    TConstArrayRef<TString> tokens,
    TVector<TTokenId>* tokensIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(
        tokens,
        tokensIds,
        Alphabet.Get(),
        GetUnitToTokenIdFuncForMMap(SourceTokenIdsToTokenId, SourceTokenIdsToTokenIdSeed),
        unknownTokenPolicy
    );
}

void TMMapBpeDictionary::Apply(
    TConstArrayRef<TStringBuf> tokens,
    TVector<TTokenId>* tokensIds,
    EUnknownTokenPolicy unknownTokenPolicy
) const {
    ApplyImpl(
        tokens,
        tokensIds,
        Alphabet.Get(),
        GetUnitToTokenIdFuncForMMap(SourceTokenIdsToTokenId, SourceTokenIdsToTokenIdSeed),
        unknownTokenPolicy
    );
}

ui32 TMMapBpeDictionary::Size() const {
    return Alphabet->Size() + BpeSize;
}

TString TMMapBpeDictionary::GetToken(TTokenId /*tokenId*/) const {
    Y_ENSURE(false, "Unsupported method");
}

ui64 TMMapBpeDictionary::GetCount(TTokenId /*tokenId*/) const {
    Y_ENSURE(false, "Unsupported method");
}

TVector<TString> TMMapBpeDictionary::GetTopTokens(ui32 /*topSize*/) const{
    Y_ENSURE(false, "Unsupported method");
}

void TMMapBpeDictionary::ClearStatsData() {
    Y_ENSURE(false, "Unsupported method");
}

TTokenId TMMapBpeDictionary::GetUnknownTokenId() const {
    return Alphabet->GetUnknownTokenId();
}

TTokenId TMMapBpeDictionary::GetEndOfSentenceTokenId() const {
    return Alphabet->GetEndOfSentenceTokenId();
}

TTokenId TMMapBpeDictionary::GetMinUnusedTokenId() const {
    return Alphabet->GetMinUnusedTokenId() + SourceTokenIdsToTokenId.size();
}

void TMMapBpeDictionary::SetAlphabet(TIntrusivePtr<TMMapDictionary> alphabet) {
    Alphabet = alphabet;
}

TIntrusiveConstPtr<TMMapDictionary> TMMapBpeDictionary::GetAlphabet() const {
    return Alphabet.Get();
}

void TMMapBpeDictionary::Save(IOutputStream* stream) const {
    stream->Write(BPE_MAGIC, BPE_MAGIC_SIZE);
    AddPadding(16 - BPE_MAGIC_SIZE, stream);

    WriteLittleEndian(BpeSize, stream);
    AddPadding(8, stream);

    const ui64 sourceTokenIdsToTokenIdSize = SourceTokenIdsToTokenId.size() * sizeof(TBucket);
    WriteLittleEndian(sourceTokenIdsToTokenIdSize, stream);
    WriteLittleEndian(SourceTokenIdsToTokenIdSeed, stream);

    stream->Write(reinterpret_cast<const char*>(SourceTokenIdsToTokenId.data()), sourceTokenIdsToTokenIdSize);
}

void TMMapBpeDictionary::Load(IInputStream* stream) {
    char magic[BPE_MAGIC_SIZE];
    stream->Read(magic, BPE_MAGIC_SIZE);
    Y_ENSURE(!std::memcmp(magic, BPE_MAGIC, BPE_MAGIC_SIZE));
    SkipPadding(16 - BPE_MAGIC_SIZE, stream);

    ReadLittleEndian(&BpeSize, stream);
    SkipPadding(8, stream);

    ui64 sourceTokenIdsToTokenIdSize;
    ReadLittleEndian(&sourceTokenIdsToTokenIdSize, stream);
    ReadLittleEndian(&SourceTokenIdsToTokenIdSeed, stream);

    SourceTokenIdsToTokenIdBuffer.resize(sourceTokenIdsToTokenIdSize / sizeof(TBucket));
    stream->Read(SourceTokenIdsToTokenIdBuffer.data(), sourceTokenIdsToTokenIdSize);
    SourceTokenIdsToTokenId = MakeArrayRef(SourceTokenIdsToTokenIdBuffer);
}

void TMMapBpeDictionary::InitFromMemory(const void* data, size_t size) {
    const ui8* ptr = reinterpret_cast<const ui8*>(data);
    Y_ENSURE(!std::memcmp(ptr, BPE_MAGIC, BPE_MAGIC_SIZE));
    ptr += 16;

    BpeSize = *reinterpret_cast<const ui64*>(ptr);
    ptr += 16;

    ui64 sourceTokenIdsToTokenIdSize = *reinterpret_cast<const ui64*>(ptr);
    ptr += 8;
    SourceTokenIdsToTokenIdSeed = *reinterpret_cast<const ui64*>(ptr);
    ptr += 8;

    const TBucket* bucketDataBegin = reinterpret_cast<const TBucket*>(ptr);
    const TBucket* bucketDataEnd = reinterpret_cast<const TBucket*>(ptr + sourceTokenIdsToTokenIdSize);
    SourceTokenIdsToTokenId = MakeArrayRef(bucketDataBegin, bucketDataEnd);
    Y_ENSURE(size == 16 + 16 + 16 + sourceTokenIdsToTokenIdSize);
}
