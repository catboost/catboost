#include "typelist.h"

static_assert(TSignedInts::THave<char>::Result, "expect TSignedInts::THave<char>::Result");
