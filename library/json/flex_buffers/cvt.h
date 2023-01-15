#pragma once

#include <util/generic/vector.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>

namespace NJson {
    using TFlexBuffersData = TVector<ui8>;

    TString FlexToString(const TFlexBuffersData& v);
    void ConvertJsonToFlexBuffers(TStringBuf input, TFlexBuffersData& result);

    inline TFlexBuffersData ConvertJsonToFlexBuffers(TStringBuf input) {
        TFlexBuffersData result;

        ConvertJsonToFlexBuffers(input, result);

        return result;
    }
}
