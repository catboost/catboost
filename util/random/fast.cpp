#include "fast.h"

#include <util/stream/input.h>

TFastRng64::TArgs::TArgs(IInputStream& entropy) {
    static_assert(sizeof(*this) == 3 * sizeof(ui64), "please, fix me");
    entropy.LoadOrFail(this, sizeof(*this));
}

template <class T>
static inline T Read(IInputStream& in) noexcept {
    T t = T();

    in.LoadOrFail(&t, sizeof(t));

    return t;
}

TFastRng32::TFastRng32(IInputStream& entropy)
    : TFastRng32(Read<ui64>(entropy), Read<ui32>(entropy))
{
}

TReallyFastRng32::TReallyFastRng32(IInputStream& entropy)
    : TReallyFastRng32(Read<ui64>(entropy))
{
}
