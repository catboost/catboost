#include "flags.h"

#include <util/stream/format.h>
#include <util/system/yassert.h>

void ::NPrivate::PrintFlags(IOutputStream& stream, ui64 value, size_t size) {
    /* Note that this function is in cpp because we need to break circular
     * dependency between TFlags and ENumberFormat. */
    stream << "TFlags(";

    switch (size) {
        case 1:
            stream << Bin(static_cast<ui8>(value), HF_FULL);
            break;
        case 2:
            stream << Bin(static_cast<ui16>(value), HF_FULL);
            break;
        case 4:
            stream << Bin(static_cast<ui32>(value), HF_FULL);
            break;
        case 8:
            stream << Bin(static_cast<ui64>(value), HF_FULL);
            break;
        default:
            Y_ABORT_UNLESS(false);
    }

    stream << ")";
}
