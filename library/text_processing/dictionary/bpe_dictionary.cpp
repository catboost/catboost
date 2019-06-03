#include "bpe_builder.h"

#include <library/streams/factory/factory.h>

#include <util/string/split.h>
#include <util/generic/queue.h>
#include <util/generic/hash_set.h>

using namespace NTextProcessing::NDictionary;
using NTextProcessing::NDictionary::EUnknownTokenPolicy;
using TUnit = std::pair<TTokenId, TTokenId>;
using TTokenToUnit = std::pair<TTokenId, TUnit>;
using TUnitQueue = TPriorityQueue<TTokenToUnit, TVector<TTokenToUnit>, std::greater<TTokenToUnit>>;

void TBpeDictionary::Load(const TString& dictionaryPath, const TString& bpePath) {
    auto dictInput = OpenInput(dictionaryPath);
    Alphabet = MakeIntrusive<TDictionary>();
    Alphabet->Load(dictInput.Get());

    auto bpeInput = OpenInput(bpePath);
    TString line;
    while (bpeInput->ReadLine(line)) {
        TBpeUnit unit;
        TString _;
        StringSplitter(line).Split('\t').Limit(4).CollectInto(&unit.Left, &unit.Right, &unit.Count, &_);
        BpeUnits.push_back(unit);
    }
    InitBpeTokens();
}

void TBpeDictionary::Save(const TString& dictionaryPath, const TString& bpePath) const {
    auto dictionaryOutput = OpenOutput(dictionaryPath);
    GetAlphabet()->Save(dictionaryOutput.Get());
    auto bpeDictionaryOutput = OpenOutput(bpePath);
    Save(bpeDictionaryOutput.Get());
}

static void AddUnit(
    const TUnit& unit,
    const THashMap<std::pair<TTokenId, TTokenId>, TTokenId>& sourceTokenIdsToTokenId,
    THashMap<TUnit, int>* unitCounts,
    TUnitQueue* minIds) {

    int countOfNewUnits = ++(*unitCounts)[unit];
    if (countOfNewUnits == 1) {
        auto unitIdIter = sourceTokenIdsToTokenId.find(unit);
        if (unitIdIter != sourceTokenIdsToTokenId.end()) {
            minIds->emplace(unitIdIter->second, unit);
        }
    }
}

static void UpdatePrevUnit(
    const TVector<TTokenId>& tokenIds,
    int idxStartInOld,
    int idxStartInNew,
    TTokenId changedToken,
    const THashMap<std::pair<TTokenId, TTokenId>, TTokenId>& sourceTokenIdsToTokenId,
    THashMap<TUnit, int>* unitCounts,
    TUnitQueue* minIds) {
    if (idxStartInNew == 0) {
        return;
    }
    TUnit oldPrevUnit{tokenIds[idxStartInNew - 1], tokenIds[idxStartInOld]};
    (*unitCounts)[oldPrevUnit]--;

    TUnit newPrevUnit{tokenIds[idxStartInNew - 1], changedToken};
    AddUnit(newPrevUnit, sourceTokenIdsToTokenId, unitCounts, minIds);
}

static void UpdateNextUnit(
    const TVector<TTokenId>& tokenIds,
    int idxStartInOld,
    TTokenId changedToken,
    const THashMap<std::pair<TTokenId, TTokenId>, TTokenId>& sourceTokenIdsToTokenId,
    THashMap<TUnit, int>* unitCounts,
    TUnitQueue* minIds) {

    if (idxStartInOld + 2 >= tokenIds.ysize()) {
        return;
    }
    TUnit oldNextUnit{tokenIds[idxStartInOld + 1], tokenIds[idxStartInOld + 2]};
    (*unitCounts)[oldNextUnit]--;

    TUnit newNextUnit{changedToken, tokenIds[idxStartInOld + 2]};
    AddUnit(newNextUnit, sourceTokenIdsToTokenId, unitCounts, minIds);
}

// TODO(annaveronika): more efficient apply.
template <typename TStringVector>
static void ApplyImpl(
    TStringVector tokens,
    TVector<TTokenId>* tokenIds,
    const TDictionary* alphabet,
    const THashMap<std::pair<TTokenId, TTokenId>, TTokenId>& sourceTokenIdsToTokenId,
    EUnknownTokenPolicy unknownTokenPolicy
) {
    tokenIds->clear();
    alphabet->Apply(tokens, tokenIds, unknownTokenPolicy);

    TPriorityQueue<TTokenToUnit, TVector<TTokenToUnit>, std::greater<TTokenToUnit>> minIds;
    THashMap<TUnit, int> unitCounts;

    for (size_t i = 0; i + 1 < tokenIds->size(); ++i) {
        auto unit = TUnit((*tokenIds)[i], (*tokenIds)[i + 1]);
        auto it = sourceTokenIdsToTokenId.find(unit);
        if (it == sourceTokenIdsToTokenId.end()) {
            continue;
        }
        auto unitId = it->second;
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
                UpdatePrevUnit(*tokenIds, i, j, bestId, sourceTokenIdsToTokenId, &unitCounts, &minIds);
                UpdateNextUnit(*tokenIds, i, bestId, sourceTokenIdsToTokenId, &unitCounts, &minIds);

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

TTokenId TBpeDictionary::Apply(TStringBuf) const {
    Y_ENSURE(false, "This method is unimplemented for TBpeDictionary.");
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
        SourceTokenIdsToTokenId,
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
        SourceTokenIdsToTokenId,
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

TString TBpeDictionary::GetBpeToken(TTokenId leftId, TTokenId rightId) const {
    if (Alphabet->GetDictionaryOptionsRef().TokenLevelType == ETokenLevelType::Word) {
        return TString::Join(GetToken(leftId), " ", GetToken(rightId));
    } else {
        Y_ASSERT(Alphabet->GetDictionaryOptionsRef().TokenLevelType == ETokenLevelType::Letter);
        return TString::Join(GetToken(leftId), GetToken(rightId));
    }
}

void TBpeDictionary::Save(IOutputStream* output) const {
    for (const auto& unit : BpeUnits) {
        *output << unit.Left << '\t' << unit.Right << '\t' << unit.Count << '\t' << GetBpeToken(unit.Left, unit.Right) << '\n';
    }
}

void TBpeDictionary::InitBpeTokens() {
    TTokenId curTokenId = GetMinTokenIdForUnits();
    for (const auto& unit : BpeUnits) {
        SourceTokenIdsToTokenId[std::pair<TTokenId, TTokenId>(unit.Left, unit.Right)] = curTokenId++;
        StringTokens.push_back(GetBpeToken(unit.Left, unit.Right));
    }
}
