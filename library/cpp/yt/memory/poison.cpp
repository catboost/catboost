#include "poison.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace {

template <char Byte0, char Byte1, char Byte2, char Byte3>
void ClobberMemory(char* __restrict__ ptr, size_t size)
{
    while (size >= 4) {
        *ptr++ = Byte0;
        *ptr++ = Byte1;
        *ptr++ = Byte2;
        *ptr++ = Byte3;
        size -= 4;
    }

    switch (size) {
        case 3:
            *ptr++ = Byte0;
            [[fallthrough]];
        case 2:
            *ptr++ = Byte1;
            [[fallthrough]];
        case 1:
            *ptr++ = Byte2;
    }
}

} // namespace

#if !defined(NDEBUG) && !defined(_asan_enabled_) && !defined(_msan_enabled_)

void PoisonUninitializedMemory(TMutableRef ref)
{
    // BAADBOBA
    ClobberMemory<'\xba', '\xad', '\xb0', '\xba'>(ref.data(), ref.size());
}

void PoisonFreedMemory(TMutableRef ref)
{
    // DEADBEEF
    ClobberMemory<'\xde', '\xad', '\xbe', '\xef'>(ref.data(), ref.size());
}

void RecycleFreedMemory(TMutableRef ref)
{
    // COOLBIBA
    ClobberMemory<'\xc0', '\x01', '\xb1', '\xba'>(ref.data(), ref.size());
}

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
