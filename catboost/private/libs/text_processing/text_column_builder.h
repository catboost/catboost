#pragma once

#include "text_dataset.h"
#include "tokenizer.h"
#include <array>
#include <util/generic/fwd.h>


namespace NCB {

    class TTextColumnBuilder {
    public:
        TTextColumnBuilder(TTokenizerPtr tokenizer, TDictionaryPtr dictionary, ui32 samplesCount)
            : Tokenizer(std::move(tokenizer))
            , Dictionary(std::move(dictionary))
            , Texts(samplesCount)
        {}

        void AddText(ui32 index, TStringBuf text);

        TVector<TText> Build();

    private:
        TTokenizerPtr Tokenizer;
        TDictionaryPtr Dictionary;

        TVector<TText> Texts;

        bool WasBuilt = false;
    };

}
