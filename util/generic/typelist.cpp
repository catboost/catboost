#include "typelist.h"

static_assert(
    TSignedInts::THave<char>::value,
    "char type in Arcadia must be signed; add -fsigned-char to compiler options");
