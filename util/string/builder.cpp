#include "builder.h"

#include <util/stream/output.h>

template <>
void Out<TStringBuilder>(TOutputStream& os, const TStringBuilder& sb) {
    os << static_cast<const TString&>(sb);
}
