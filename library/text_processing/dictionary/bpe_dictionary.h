#pragma once

#include "dictionary.h"
#include "frequency_based_dictionary.h"
#include "mmap_frequency_based_dictionary.h"
#include "mmap_hash_table.h"

#include <util/generic/vector.h>
#include <util/stream/output.h>

namespace NTextProcessing::NDictionary {

    class TBpeDictionaryBuilder;
    class TMMapBpeDictionary;

    class TBpeDictionary final : public IDictionary, public TMoveOnly {
    public:
        TBpeDictionary() = default;

        explicit TBpeDictionary(TIntrusivePtr<TDictionary> alphabet);

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

        TTokenId GetUnknownTokenId() const override;
        TTokenId GetEndOfSentenceTokenId() const override;
        TTokenId GetMinUnusedTokenId() const override;

        void SetAlphabet(TIntrusivePtr<TDictionary> alphabet);
        TIntrusiveConstPtr<TDictionary> GetAlphabet() const;

        // These methods save/load only bpe data.
        // The alphabet must be saved/loaded separately or use next two methods.
        void Save(IOutputStream* stream) const override;
        void Load(IInputStream* stream);

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
        friend class NTextProcessing::NDictionary::TMMapBpeDictionary;

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

    class TMMapBpeDictionary final : public IDictionary, public TMoveOnly {
    public:
        TMMapBpeDictionary() = default;
        explicit TMMapBpeDictionary(TIntrusivePtr<TBpeDictionary> bpeDictionary);
        explicit TMMapBpeDictionary(TIntrusivePtr<TMMapDictionary> alphabet);
        TMMapBpeDictionary(TIntrusivePtr<TMMapDictionary> alphabet, const void* data, size_t size);

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

        TString GetToken(TTokenId /*tokenId*/) const override;
        ui64 GetCount(TTokenId /*tokenId*/) const override;
        TVector<TString> GetTopTokens(ui32 /*topSize*/) const override;

        void ClearStatsData() override;

        TTokenId GetUnknownTokenId() const override;
        TTokenId GetEndOfSentenceTokenId() const override;
        TTokenId GetMinUnusedTokenId() const override;

        void SetAlphabet(TIntrusivePtr<TMMapDictionary> alphabet);
        TIntrusiveConstPtr<TMMapDictionary> GetAlphabet() const;

        // These methods save/load only bpe data. The alphabet must be saved/loaded separately.
        void Save(IOutputStream* stream) const override;
        void Load(IInputStream* stream);

        // These method initializes only bpe data. The alphabet must be initialized separately.
        void InitFromMemory(const void* data, size_t size);

    private:
        TIntrusivePtr<TMMapDictionary> Alphabet;
        ui64 BpeSize = 0;
        TVector<TBucket> SourceTokenIdsToTokenIdBuffer;
        TConstArrayRef<TBucket> SourceTokenIdsToTokenId;
        ui64 SourceTokenIdsToTokenIdSeed = 0;
    };

}

