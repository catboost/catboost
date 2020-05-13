#include "special_tokens.h"

#include <library/cpp/containers/comptrie/set.h>

#include <util/generic/singleton.h>

namespace {
    extern "C" {
        extern const unsigned char SpecialTokens[];
        extern const ui32 SpecialTokensSize;
    }

    class TSpecialTokensSet: public TCompactTrieSet<wchar16> {
    public:
        TSpecialTokensSet(): TCompactTrieSet<wchar16>(reinterpret_cast<const char*>(SpecialTokens), SpecialTokensSize)
        {
        }
    };

    auto SpecialTokensSet = Singleton<TSpecialTokensSet>();
}

size_t GetSpecialTokenLength(const wchar16* text, size_t maxLen) {
    size_t resultLen = 0;
    SpecialTokensSet->FindLongestPrefix(text, maxLen, &resultLen);
    return resultLen;
}
