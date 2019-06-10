#include "bpe_builder.h"

#include "util.h"

#include <library/containers/dense_hash/dense_hash.h>
#include <library/containers/heap_dict/heap_dict.h>

#include <util/generic/array_ref.h>

using NTextProcessing::NDictionary::EEndOfWordTokenPolicy;
using namespace NTextProcessing::NDictionary;

template <typename T>
static void AddToken(
    const T& token,
    bool skipUnknown,
    const TDictionary& alphabet,
    TVector<TVector<TTokenId>>* lines) {

    auto tokenId = alphabet.Apply(token);
    if (tokenId != alphabet.GetUnknownTokenId()) {
        lines->back().push_back(tokenId);
    } else {
        if (skipUnknown) {
            return;
        }
        lines->back().push_back(tokenId);
    }
}

template <typename TStringVector>
static void AddImpl(
    TStringVector tokens,
    ui64 weight,
    bool skipUnknown,
    const TDictionary& alphabet,
    TVector<TVector<TTokenId>>* lines,
    TVector<ui64>* counts
) {
    const TTokenId eosId = alphabet.GetEndOfSentenceTokenId();
    const auto& alphabetOptins = alphabet.GetDictionaryOptionsRef();
    const auto tokenLevelType = alphabetOptins.TokenLevelType;

    const bool addEndOfWordToken = alphabetOptins.EndOfWordTokenPolicy == EEndOfWordTokenPolicy::Insert;
    const bool addEndOfSentenceToken = alphabetOptins.EndOfSentenceTokenPolicy == EEndOfSentenceTokenPolicy::Insert;

    lines->emplace_back();
    counts->push_back(weight);

    if (tokenLevelType == ETokenLevelType::Letter) {
        auto addTokenFunc = [&] (TStringBuf character) {
            AddToken(character, skipUnknown, alphabet, lines);
        };
        ApplyFuncToLetterNGrams(tokens, /*gramOrder*/1, addEndOfWordToken, addTokenFunc);
    } else {
        Y_ASSERT(tokenLevelType == ETokenLevelType::Word);
        for (const auto& token : tokens) {
            AddToken(token, skipUnknown, alphabet, lines);
        }
    }

    if (lines->back().empty()) {
        lines->pop_back();
        counts->pop_back();
    } else if (tokenLevelType == ETokenLevelType::Word && addEndOfSentenceToken) {
        lines->back().push_back(eosId);
    }
}

void TBpeDictionaryBuilder::Add(TConstArrayRef<TStringBuf> tokens, ui64 weight) {
    AddImpl(tokens, weight, SkipUnknown, *Alphabet, &Lines, &Counts);
}

void TBpeDictionaryBuilder::Add(TConstArrayRef<TString> tokens, ui64 weight) {
    AddImpl(tokens, weight, SkipUnknown, *Alphabet, &Lines, &Counts);
}

TIntrusivePtr<TBpeDictionary> TBpeDictionaryBuilder::FinishBuilding() {
    Y_ENSURE(!IsBuildingFinish, "FinishBuilding method should be called only once.");
    IsBuildingFinish = true;
    CalcMostFrequentUnits();
    return new TBpeDictionary(Alphabet, std::move(ResultingBpeUnits));
}

void TBpeDictionaryBuilder::CalcMostFrequentUnits() {
    ResultingBpeUnits.clear();

    using TPair = std::pair<TTokenId, TTokenId>;
    struct TPairStat {
        ui64 Count = 0;
        TTokenId SmallerTokenId = 0;
        TTokenId LargerTokenId = 0;
        // ui32 is used for lower memory consumption
        TDenseHashSet<ui32, THash<ui32>, /*maxLoadFactor*/75, /*logInitSize*/0> SrcStrIds{/*emptyKey*/Max<ui32>()};

        bool operator<(const TPairStat& other) const {
            return std::tie(Count, other.SmallerTokenId, other.LargerTokenId) < std::tie(other.Count, SmallerTokenId, LargerTokenId);
        }
    };
    using TPairStats = THeapDict<TPair, TPairStat>;

    Cerr << "Preparing stats..." << Endl;
    TPairStats pairStats;
    for (size_t i = 0; i < Lines.size(); ++i) {
        const auto& line = Lines[i];
        ui64 count = Counts[i];
        for (size_t j = 0; j + 1 < line.size(); ++j) {
            TPair pair(line[j], line[j + 1]);
            auto& stat = pairStats[pair];
            stat.SmallerTokenId = Min(pair.first, pair.second);
            stat.LargerTokenId = Max(pair.first, pair.second);
            stat.Count += count;
            stat.SrcStrIds.Insert(i);
        }
    }

    TTokenId newTokenId = Alphabet->GetMinUnusedTokenId();

    Cerr << "Training..." << Endl;
    ResultingBpeUnits.reserve(NumUnits);
    for (size_t iter = 0; iter < NumUnits; ++iter, ++newTokenId) {
        if (pairStats.empty()) {
            Cerr << "Did not manage to build " << NumUnits << " units!" << Endl;
            break;
        }

        const auto& best = pairStats.top();
        TPair bestPair = best.first;
        ui64 bestCount = best.second.Count;
        ResultingBpeUnits.emplace_back(TBpeDictionary::TBpeUnit{bestPair.first, bestPair.second, bestCount});

        auto srcStrIds = best.second.SrcStrIds;
        for (size_t strId : srcStrIds) {
            auto& line = Lines[strId];
            ui64 lineCount = Counts[strId];
            for (size_t i = line.size() - 1; i >= 1;) {
                TPair pair(line[i - 1], line[i]);
                if (pair == bestPair) {
                    if (i - 1) {
                        TPair left(line[i - 2], line[i - 1]);
                        auto it = pairStats.find(left);
                        it->second.Count -= lineCount;
                        if (it->second.Count == 0) {
                            pairStats.erase(it);
                        }
                    }
                    if (i + 1 < line.size()) {
                        TPair right(line[i], line[i + 1]);
                        auto it = pairStats.find(right);
                        it->second.Count -= lineCount;
                        if (it->second.Count == 0) {
                            pairStats.erase(it);
                        }
                    }
                    line[i - 1] = newTokenId;
                    line.erase(line.begin() + i);
                    if (i - 1) {
                        TPair left(line[i - 2], line[i - 1]);
                        auto& stat = pairStats[left];
                        stat.Count += lineCount;
                        stat.SrcStrIds.Insert(strId);
                    }
                    if (i < line.size()) {
                        TPair right(line[i - 1], line[i]);
                        auto& stat = pairStats[right];
                        stat.Count += lineCount;
                        stat.SrcStrIds.Insert(strId);
                    }
                    i -= Min<ui64>(i, 2);
                } else {
                    --i;
                }
            }
        }
        pairStats.erase(bestPair);
    }
}
