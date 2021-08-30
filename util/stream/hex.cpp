#include "hex.h"

#include "output.h"
#include <util/string/hex.h>

void HexEncode(const void* in, size_t len, IOutputStream& out) {
    static const size_t NUM_OF_BYTES = 32;
    char buffer[NUM_OF_BYTES * 2];

    auto current = static_cast<const char*>(in);
    for (size_t take = 0; len; current += take, len -= take) {
        take = Min(NUM_OF_BYTES, len);
        HexEncode(current, take, buffer);
        out.Write(buffer, take * 2);
    }
}

void HexDecode(const void* in, size_t len, IOutputStream& out) {
    Y_ENSURE(!(len & 1), TStringBuf("Odd buffer length passed to HexDecode"));

    static const size_t NUM_OF_BYTES = 32;
    char buffer[NUM_OF_BYTES];

    auto current = static_cast<const char*>(in);
    for (size_t take = 0; len; current += take, len -= take) {
        take = Min(NUM_OF_BYTES * 2, len);
        HexDecode(current, take, buffer);
        out.Write(buffer, take / 2);
    }
}
