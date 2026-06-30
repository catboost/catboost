#include <library/cpp/string_utils/base64/base64.h>

#include <util/system/types.h>
#include <util/system/yassert.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    const TStringBuf example{reinterpret_cast<const char*>(data), size};
    const auto converted = Base64Decode(Base64Encode(example));

    Y_ABORT_UNLESS(example == converted);

    return 0;
}
