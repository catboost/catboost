#include "bpe_builder.h"

using NTextProcessing::NDictionary::EEndOfWordTokenPolicy;
using namespace NTextProcessing::NDictionary;

template <typename T>
static void AddToken(
    const T& token,
    bool skipUnknown,
    const TDictionary& alphabet,
    TEraseList<TTokenId>* tokenIdsList
) {
    auto tokenId = alphabet.Apply(token);
    if (tokenId != alphabet.GetUnknownTokenId()) {
        tokenIdsList->PushToken(tokenId);
    } else {
        if (skipUnknown) {
            return;
        }
        tokenIdsList->PushToken(tokenId);
    }
}

template <typename TStringVector>
static void AddImpl(
    TStringVector tokens,
    ui64 weight,
    bool skipUnknown,
    const TDictionary& alphabet,
    TVector<TEraseList<TTokenId>>* tokenIdsLists,
    TPairStats* pairStats,
    TVector<ui64>* counts
) {
    const TTokenId eosId = alphabet.GetEndOfSentenceTokenId();
    const auto& alphabetOptins = alphabet.GetDictionaryOptionsRef();
    const auto tokenLevelType = alphabetOptins.TokenLevelType;

    const bool addEndOfWordToken = alphabetOptins.EndOfWordTokenPolicy == EEndOfWordTokenPolicy::Insert;
    const bool addEndOfSentenceToken = alphabetOptins.EndOfSentenceTokenPolicy == EEndOfSentenceTokenPolicy::Insert;

    tokenIdsLists->emplace_back();
    counts->push_back(weight);

    if (tokenLevelType == ETokenLevelType::Letter) {
        auto addTokenFunc = [&] (TStringBuf character) {
            AddToken(character, skipUnknown, alphabet, &tokenIdsLists->back());
        };
        ApplyFuncToLetterNGrams(tokens, /*gramOrder*/1, addEndOfWordToken, addTokenFunc);
    } else {
        Y_ASSERT(tokenLevelType == ETokenLevelType::Word);
        for (const auto& token : tokens) {
            AddToken(token, skipUnknown, alphabet, &tokenIdsLists->back());
        }
    }

    if (tokenIdsLists->back().Empty()) {
        tokenIdsLists->pop_back();
        counts->pop_back();
        return;
    }
    if (tokenLevelType == ETokenLevelType::Word && addEndOfSentenceToken) {
        tokenIdsLists->back().PushToken(eosId);
    }

    for (int i = (int)tokenIdsLists->back().Size() - 2; i >= 0; --i) {
        TPair pair = tokenIdsLists->back().GetPair(i);
        TPairStat& stat = (*pairStats)[pair];
        stat.Pair = pair;
        stat.Count += weight;
        stat.Positions.emplace_back(std::make_pair((int)tokenIdsLists->size() - 1, i));
    }
}

void TBpeDictionaryBuilder::Add(TConstArrayRef<TStringBuf> tokens, ui64 weight) {
    AddImpl(tokens, weight, SkipUnknown, *Alphabet, &TokenIdsLists, &PairStats, &Counts);
}

void TBpeDictionaryBuilder::Add(TConstArrayRef<TString> tokens, ui64 weight) {
    AddImpl(tokens, weight, SkipUnknown, *Alphabet, &TokenIdsLists, &PairStats, &Counts);
}

TIntrusivePtr<TBpeDictionary> TBpeDictionaryBuilder::FinishBuilding() {
    Y_ENSURE(!IsBuildingFinish, "FinishBuilding method should be called only once.");
    IsBuildingFinish = true;
    CalcMostFrequentUnits();
    return new TBpeDictionary(Alphabet, std::move(ResultingBpeUnits));
}

void TBpeDictionaryBuilder::CalcMostFrequentUnits() {
    ResultingBpeUnits.clear();
    TTokenId newTokenId = Alphabet->GetMinUnusedTokenId();

    Cerr << "Training..." << Endl;
    ResultingBpeUnits.reserve(NumUnits);
    for (size_t iter = 0; iter < NumUnits; ++iter, ++newTokenId) {
        if (PairStats.empty()) {
            Cerr << "Did not manage to build " << NumUnits << " units!" << Endl;
            break;
        }

        const auto& best = PairStats.top();
        TPair bestPair = best.first;
        ui64 bestCount = best.second.Count;
        ResultingBpeUnits.emplace_back(TBpeDictionary::TBpeUnit{bestPair.first, bestPair.second, bestCount});

        for (auto position : best.second.Positions) {
            int lineId = position.first;
            auto& tokenIdsList = TokenIdsLists[lineId];
            int firstPosition = position.second;
            if (
                !tokenIdsList.IsValidElement(firstPosition) ||
                tokenIdsList.IsLastElement(firstPosition) ||
                tokenIdsList.GetPair(firstPosition) != bestPair
            ) {
                continue;
            }
            ui64 count = Counts[lineId];

            const auto removePair = [&](int position) {
                TPair pair = tokenIdsList.GetPair(position);
                auto it = PairStats.find(pair);
                it->second.Count -= count;
                if (it->second.Count == 0) {
                    PairStats.erase(it);
                }
            };

            if (!tokenIdsList.IsFirstElement(firstPosition)) {
                int prevPosition = tokenIdsList.GetPrevPosition(firstPosition);
                removePair(prevPosition);
            }
            int secondPosition = tokenIdsList.GetNextPosition(firstPosition);
            if (!tokenIdsList.IsLastElement(secondPosition)) {
                removePair(secondPosition);
            }

            tokenIdsList.Erase(secondPosition);
            tokenIdsList.UpdateToken(firstPosition, newTokenId);

            const auto addPair = [&](int position) {
                TPair pair = tokenIdsList.GetPair(position);
                auto& stat = PairStats[pair];
                stat.Pair = pair;
                stat.Count += count;
                stat.Positions.emplace_back(lineId, position);
            };

            if (!tokenIdsList.IsLastElement(firstPosition)) {
                addPair(firstPosition);
            }
            if (!tokenIdsList.IsFirstElement(firstPosition)) {
                int prevPosition = tokenIdsList.GetPrevPosition(firstPosition);
                addPair(prevPosition);
            }
        }
        PairStats.erase(bestPair);
    }
}
