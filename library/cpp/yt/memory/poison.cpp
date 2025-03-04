#include "poison.h"

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace {

template <char Byte0, char Byte1, char Byte2, char Byte3, char Byte4, char Byte5, char Byte6, char Byte7>
void ClobberMemory(char* __restrict__ ptr, size_t size)
{
    while (size >= 8) {
        *ptr++ = Byte0;
        *ptr++ = Byte1;
        *ptr++ = Byte2;
        *ptr++ = Byte3;
        *ptr++ = Byte4;
        *ptr++ = Byte5;
        *ptr++ = Byte6;
        *ptr++ = Byte7;
        size -= 8;
    }

    switch (size) {
        case 7:
            *ptr++ = Byte0;
            [[fallthrough]];
        case 6:
            *ptr++ = Byte1;
            [[fallthrough]];
        case 5:
            *ptr++ = Byte2;
            [[fallthrough]];
        case 4:
            *ptr++ = Byte3;
            [[fallthrough]];
        case 3:
            *ptr++ = Byte4;
            [[fallthrough]];
        case 2:
            *ptr++ = Byte5;
            [[fallthrough]];
        case 1:
            *ptr++ = Byte6;
    }
}

} // namespace

#if !defined(NDEBUG) && !defined(_asan_enabled_) && !defined(_msan_enabled_)
void PoisonMemory(TMutableRef ref)
{
    ClobberMemory<'d', 'e', 'a', 'd', 'b', 'e', 'e', 'f'>(ref.data(), ref.size());
}

void UnpoisonMemory(TMutableRef ref)
{
    ClobberMemory<'c', 'a', 'f', 'e', 'b', 'a', 'b', 'e'>(ref.data(), ref.size());
}
#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
