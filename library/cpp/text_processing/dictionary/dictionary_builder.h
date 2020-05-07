#pragma once

#include "frequency_based_dictionary.h"
#include "options.h"

#include <util/generic/array_ref.h>

namespace NTextProcessing::NDictionary {
    class IDictionaryBuilderImpl;

    class TDictionaryBuilder: public TMoveOnly {
    public:
        TDictionaryBuilder(TDictionaryBuilder&&);
        ~TDictionaryBuilder();

        TDictionaryBuilder(
            const TDictionaryBuilderOptions& dictionaryBuilderOptions,
            const TDictionaryOptions& dictionaryOptions
        );

        /*
         * This method is intended for token adding to dictionary.
         * Example:
         *      dictionaryBuilder.Add("apple");
         * */
        void Add(TStringBuf token, ui64 weight = 1);

        /*
         * These methods are intended for token adding to dictionary.
         * Example:
         *      TVector<TString> firstSentence = {"he", "likes", "apples"};
         *      dictionaryBuilder.Add(firstSentence);
         *      TVector<TString> secondSentence = {"she", "does", "not", "like", "winter"};
         *      dictionaryBuilder.Add(secondSentence);
         * */
        void Add(TConstArrayRef<TString> tokens, ui64 weight = 1);
        void Add(TConstArrayRef<TStringBuf> tokens, ui64 weight = 1);

        TIntrusivePtr<TDictionary> FinishBuilding();

    private:
        THolder<IDictionaryBuilderImpl> DictionaryBuilderImpl;
    };
}
