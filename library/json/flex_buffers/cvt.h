#pragma once

#include <util/generic/vector.h>
#include <util/generic/strbuf.h>

namespace NJson {
    void ConvertJsonToFlexBuffers(TStringBuf input, TVector<ui8>& result);
}
