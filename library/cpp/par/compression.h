#pragma once

#include <util/generic/vector.h>

namespace NPar {
    // pack small packets, for packed add signature
    void QuickLZCompress(TVector<char>* dst);
    void QuickLZDecompress(TVector<char>* dst);
}
