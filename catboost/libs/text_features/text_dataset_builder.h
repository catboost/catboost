#pragma once

#include "tokenizer.h"
#include "text_dataset.h"
#include "dictionary.h"
#include <array>
#include <util/generic/fwd.h>


namespace NCB {

    using IDictionary = NTextProcessing::NDictionary::IDictionary;
    using TDictionaryPtr = TIntrusivePtr<IDictionary>;

    TText TokensToText(const IDictionary& dictionary, TConstArrayRef<TString> tokens);

    inline TText TokenToText(const IDictionary& dictionary, TString token) {
        std::array<TString, 1> tmp{token};
        return TokensToText(dictionary, MakeConstArrayRef(tmp));
    }


    class TTextDataSetBuilder {
    public:
        TTextDataSetBuilder(TTokenizerPtr tokenizer,
                            TDictionaryPtr dictionary)
        : Tokenizer(tokenizer)
        , Dictionary(dictionary) {

        }

        void AddText(TStringBuf text);

        TIntrusivePtr<TTextDataSet> Build();

    private:
        TTokenizerPtr Tokenizer;
        TDictionaryPtr Dictionary;

        TVector<TText> Texts;

        bool WasBuilt = false;
    };

}
