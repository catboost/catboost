#include "bpe_builder.h"

#include <library/streams/factory/factory.h>

#include <util/string/split.h>

using namespace NTextProcessing::NDictionary;
using NTextProcessing::NDictionary::EUnknownTokenPolicy;

void TBpeDictionary::Load(const TString& dictionaryPath, const TString& bpePath) {
    auto dictInput = OpenInput(dictionaryPath);
    Alphabet = MakeIntrusive<TDictionary>();
    Alphabet->Load(dictInput.Get());

    auto bpeInput = OpenInput(bpePath);
    TString line;
    while (bpeInput->ReadLine(line)) {
        TBpeUnit unit;
        TString _;
        StringSplitter(line).SplitLimited('\t', 4).CollectInto(&unit.Left, &unit.Right, &unit.Count, &_);
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

// TODO(annaveronika): more efficient apply.
template <typename TStringVector>
static void ApplyImpl(
    TStringVector tokens,
    TVector<TTokenId>* tokenIds,
    const TDictionary* alphabet,
    const THashMap<std::pair<TTokenId, TTokenId>, TTokenId>& sourceTokenIdsToTokenId,
    TTokenId minUnusedTokenId,
    EUnknownTokenPolicy unknownTokenPolicy
) {

    tokenIds->clear();
    alphabet->Apply(tokens, tokenIds, unknownTokenPolicy);

    while (tokenIds->size() > 1) {
        auto minTokenWithMerge = minUnusedTokenId;
        std::pair<TTokenId, TTokenId> bestUnit;
        for (size_t i = 0; i + 1 < tokenIds->size(); ++i) {
            auto unit = std::pair<TTokenId, TTokenId>((*tokenIds)[i], (*tokenIds)[i + 1]);
            auto it = sourceTokenIdsToTokenId.find(unit);
            if (it == sourceTokenIdsToTokenId.end()) {
                continue;
            }
            if (it->second < minTokenWithMerge) {
                minTokenWithMerge = it->second;
                bestUnit = unit;
            }
        }
        if (minTokenWithMerge == minUnusedTokenId) {
            break;
        }

        size_t i = 0, j = 0;
        while (i < tokenIds->size()) {
            if (i + 1 < tokenIds->size() && bestUnit.first == (*tokenIds)[i] && bestUnit.second == (*tokenIds)[i + 1]) {
                (*tokenIds)[j] = minTokenWithMerge;
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
        GetMinUnusedTokenId(),
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
        GetMinUnusedTokenId(),
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
