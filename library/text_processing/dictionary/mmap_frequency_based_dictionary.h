#pragma once

#include "dictionary.h"
#include "options.h"

#include <util/memory/blob.h>

namespace NTextProcessing::NDictionary {

    class IMMapDictionaryImpl;

    class TMMapDictionary final : public IDictionary {
    public:
        TMMapDictionary();
        TMMapDictionary(TMMapDictionary&&);
        ~TMMapDictionary();
        explicit TMMapDictionary(THolder<IMMapDictionaryImpl> dictionaryImpl);

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

        void InitFromMemory(const void* data, size_t size);

    private:
        THolder<IMMapDictionaryImpl> DictionaryImpl;
    };

}
