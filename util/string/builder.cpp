#include "builder.h"

#include <util/stream/output.h>

template <>
void Out<TStringBuilder>(IOutputStream& os, const TStringBuilder& sb) {
    os << static_cast<const TString&>(sb);
}
