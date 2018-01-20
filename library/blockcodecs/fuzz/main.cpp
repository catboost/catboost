#include <library/blockcodecs/codecs.h>
#include <library/blockcodecs/stream.h>

#include <util/stream/mem.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    TMemoryInput mi(data, size);

    try {
        NBlockCodecs::TDecodedInput(&mi).ReadAll();
    } catch (...) {
    }

    return 0; // Non-zero return values are reserved for future use.
}
