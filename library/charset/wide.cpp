#include "wide.h"

bool CanBeEncoded(TWtringBuf text, ECharset encoding) {
    const size_t LEN = 16;
    const size_t BUFSIZE = LEN * 4;
    char encodeBuf[BUFSIZE];
    wchar16 decodeBuf[BUFSIZE];

    while (!text.empty()) {
        TWtringBuf src = text.NextTokAt(LEN);
        TStringBuf encoded = NDetail::NBaseOps::Recode(src, encodeBuf, encoding);
        TWtringBuf decoded = NDetail::NBaseOps::Recode(encoded, decodeBuf, encoding);
        if (decoded != src)
            return false;
    }

    return true;
}
