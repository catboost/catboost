#pragma once

#include <catboost/private/libs/options/enums.h>
#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/ptr.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>

namespace NCB {

    //TODO(noxoomo, nikitxskv): move to library after text-processing tokenizer will be available
    class ITokenizer : public TThrRefBase {
    public:
        virtual void Tokenize(TStringBuf inputString, TVector<TStringBuf>* tokens) const = 0;
    };

    using TTokenizerPtr = TIntrusivePtr<ITokenizer>;

    TVector<TVector<TStringBuf>> Tokenize(TConstArrayRef<TStringBuf> textFeature, const TTokenizerPtr& tokenizer);

    TTokenizerPtr CreateTokenizer(ETokenizerType tokenizerType = ETokenizerType::Naive);
}


