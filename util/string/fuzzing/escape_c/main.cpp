#include <util/generic/string.h>
#include <util/string/escape.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* const data, const size_t size) {
    const TString src(reinterpret_cast<const char*>(data), size);
    const auto escaped = EscapeC(src);
    const auto dst = UnescapeC(escaped);

    Y_VERIFY(src == dst);
    return 0;
}
