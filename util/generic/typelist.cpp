#include "typelist.h"

static_assert(TSignedInts::THave<char>::value, "expect TSignedInts::THave<char>::value");
