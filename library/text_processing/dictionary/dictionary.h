#pragma once

#include "types.h"

#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NTextProcessing::NDictionary {

    class IDictionary : public TThrRefBase {
    public:
        /*
         * This method is intended for dictionary application.
         * Example:
         *      TTokenId tokenId = dictionary->Apply("apple");
         * */
        virtual TTokenId Apply(TStringBuf token) const = 0;

        /*
         * These methods are intended for dictionary application.
         * Example:
         *      TVector<TString> firstSentence = {"he", "likes", "apples"};
         *      TVector<TTokenId> firstTokenIds;
         *      dictionary->Apply(firstSentence, &firstTokenIds);
         *
         *      TVector<TString> secondSentence = {"she", "does", "not", "like", "winter"};
         *      TVector<TTokenId> secondTokenIds;
         *      dictionary->Apply(secondSentence, &secondTokenIds);
         * */
        virtual void Apply(
            TConstArrayRef<TString> tokens,
            TVector<TTokenId>* tokensIds,
            EUnknownTokenPolicy unknownTokenPolicy = EUnknownTokenPolicy::Skip
        ) const = 0;
        virtual void Apply(
            TConstArrayRef<TStringBuf> tokens,
            TVector<TTokenId>* tokenIds,
            EUnknownTokenPolicy unknownTokenPolicy = EUnknownTokenPolicy::Skip
        ) const = 0;

        /*
         * Dictionary size.
         * */
        virtual ui32 Size() const = 0;

        /*
         * Get token by tokenId.
         *
         * Example:
         * auto tokenId = dictionary->Apply("cat");
         * auto token = dictionary->GetToken(tokenId);
         * */
        virtual TString GetToken(TTokenId tokenId) const = 0;

        /*
         * Get tokens by tokenIds.
         * */
        void GetTokens(TConstArrayRef<TTokenId> tokenIds, TVector<TString>* tokens) const;

        /*
         * Get token count by tokenId.
         *
         * Example:
         * auto tokenId = dictionary->Apply("cat");
         * auto tokenCount = dictionary->GetCount(tokenId);
         * */
        virtual ui64 GetCount(TTokenId tokenId) const = 0;

        /*
         * Get top tokens.
         * */
        virtual TVector<TString> GetTopTokens(ui32 topSize = 10) const = 0;

        /*
         * Clear additional data.
         * It helps to reduce dictionary memory costs.
         * */
        virtual void ClearStatsData() = 0;

        /*
         * Get Unknown tokenId.
         * */
        virtual TTokenId GetUnknownTokenId() const = 0;

        /*
         * Get EndOfSentence tokenId.
         * */
        virtual TTokenId GetEndOfSentenceTokenId() const = 0;

        /*
         * Get min unused tokenId.
         * */
        virtual TTokenId GetMinUnusedTokenId() const = 0;

        /*
         * Serialize dictionary to stream.
         *
         * Example:
         * TFileOutput output("serialized_dictionary.txt");
         * dictionary->Save(&output);
         * */
        virtual void Save(IOutputStream* stream) const = 0;

        /*
         * Factory method for loading a dictionary from a stream.
         *
         * Example:
         * TFileInput input("serialized_dictionary.txt");
         * const auto dictionary = IDictionary::Load(&input);
         * */
        static TIntrusivePtr<IDictionary> Load(IInputStream* stream);

        virtual ~IDictionary() = default;
    };

}
