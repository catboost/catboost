#include "ztstrbuf.h"

#include <util/stream/output.h>

template <>
void Out<TZtStringBuf>(IOutputStream& os, const TZtStringBuf& sb) {
    os << static_cast<const TStringBuf&>(sb);
}
