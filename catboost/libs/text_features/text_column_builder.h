#pragma once

#include "text_dataset.h"
#include "tokenizer.h"
#include <array>
#include <util/generic/fwd.h>


namespace NCB {

    using IDictionary = NTextProcessing::NDictionary::IDictionary;
    using TDictionaryPtr = TIntrusivePtr<IDictionary>;

    TText TokensToText(const IDictionary& dictionary, TConstArrayRef<TStringBuf> tokens);

    inline TText TokenToText(const IDictionary& dictionary, TStringBuf token) {
        std::array<TStringBuf, 1> tmp{token};
        return TokensToText(dictionary, MakeConstArrayRef(tmp));
    }

    class TTextColumnBuilder {
    public:
        TTextColumnBuilder(TTokenizerPtr tokenizer, TDictionaryPtr dictionary, ui32 samplesCount)
            : Tokenizer(std::move(tokenizer))
            , Dictionary(std::move(dictionary))
            , Texts(samplesCount)
        {}

        void AddText(ui32 index, TStringBuf text);

        TTextColumn Build();

    private:
        TTokenizerPtr Tokenizer;
        TDictionaryPtr Dictionary;

        TVector<TText> Texts;

        bool WasBuilt = false;
    };

}
