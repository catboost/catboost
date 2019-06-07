#pragma once

#include "dictionary.h"
#include "frequency_based_dictionary.h"

#include <util/generic/vector.h>
#include <util/stream/output.h>

namespace NTextProcessing::NDictionary {
    class TBpeDictionaryBuilder;

    class TBpeDictionary final : public IDictionary {
    public:
        TBpeDictionary() = default;

        TTokenId Apply(TStringBuf token) const override;

        void Apply(
            TConstArrayRef<TString> tokens,
            TVector<TTokenId>* tokensIds,
            EUnknownTokenPolicy unknownTokenPolicy = EUnknownTokenPolicy::Skip
        ) const override;

        void Apply(
            TConstArrayRef<TStringBuf> tokens,
            TVector<TTokenId>* tokensIds,
            EUnknownTokenPolicy unknownTokenPolicy = EUnknownTokenPolicy::Skip
        ) const override;

        ui32 Size() const override;

        TString GetToken(TTokenId tokenId) const override;
        ui64 GetCount(TTokenId tokenId) const override;
        TVector<TString> GetTopTokens(ui32 topSize = 10) const override;

        void ClearStatsData() override;

        TTokenId GetUnknownTokenId() const override {
            return Alphabet->GetUnknownTokenId();
        }

        TTokenId GetEndOfSentenceTokenId() const override {
            return Alphabet->GetEndOfSentenceTokenId();
        }

        TTokenId GetMinUnusedTokenId() const override {
            return Alphabet->GetMinUnusedTokenId() + BpeUnits.size();
        }

        TIntrusiveConstPtr<TDictionary> GetAlphabet() const {
            return Alphabet.Get();
        }

        void Save(IOutputStream* output) const override;

        // TODO(annaveronika): Allow loading without GetToken functionality (less memory).
        void Load(const TString& dictionaryPath, const TString& bpePath);
        void Save(const TString& dictionaryPath, const TString& bpePath) const;

    private:
        struct TBpeUnit {
            TTokenId Left;
            TTokenId Right;
            ui64 Count;
        };

        friend class NTextProcessing::NDictionary::TBpeDictionaryBuilder;

        explicit TBpeDictionary(TIntrusivePtr<TDictionary> alphabet, TVector<TBpeUnit> bpeUnits)
            : Alphabet(alphabet)
            , BpeUnits(std::move(bpeUnits)) {
            InitBpeTokens();
        }

        TTokenId GetMinTokenIdForUnits() const {
            return Alphabet->GetMinUnusedTokenId();
        }

        TString GetBpeToken(TTokenId leftId, TTokenId rightId) const ;

        void InitBpeTokens();

        TIntrusivePtr<TDictionary> Alphabet;
        TVector<TBpeUnit> BpeUnits;
        TVector<TString> StringTokens;
        THashMap<std::pair<TTokenId, TTokenId>, TTokenId> SourceTokenIdsToTokenId;
    };
}

