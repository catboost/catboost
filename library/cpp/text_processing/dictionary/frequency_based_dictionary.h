#pragma once

#include "dictionary.h"
#include "options.h"

namespace NTextProcessing::NDictionary {

    class IDictionaryImpl;

    class TDictionary final : public IDictionary, public TMoveOnly {
    public:
        TDictionary();
        TDictionary(TDictionary&&);
        ~TDictionary();
        explicit TDictionary(THolder<IDictionaryImpl> dictionaryImpl);

        TTokenId Apply(const TStringBuf token) const override;

        void Apply(
            TConstArrayRef<TString> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy = EUnknownTokenPolicy::Skip
        ) const override;
        void Apply(
            TConstArrayRef<TStringBuf> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy = EUnknownTokenPolicy::Skip
        ) const override;

        ui32 Size() const override;

        TString GetToken(TTokenId tokenId) const override;
        ui64 GetCount(TTokenId tokenId) const override;
        TVector<TString> GetTopTokens(ui32 topSize = 10) const override;

        void ClearStatsData() override;

        TTokenId GetUnknownTokenId() const override;
        TTokenId GetEndOfSentenceTokenId() const override;
        TTokenId GetMinUnusedTokenId() const override;

        const TDictionaryOptions& GetDictionaryOptionsRef() const;

        void Save(IOutputStream* stream) const override;
        void Load(IInputStream* stream);

    private:
        friend class TMMapDictionary;

        THolder<IDictionaryImpl> DictionaryImpl;
    };

}
