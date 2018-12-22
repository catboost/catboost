#include "unescape.h"

#include <util/string/escape.h>

TStringBuf UnescapeJsonUnicode(TStringBuf data, char* scratch) {
    return TStringBuf(scratch, UnescapeC(data.data(), data.size(), scratch));
}
