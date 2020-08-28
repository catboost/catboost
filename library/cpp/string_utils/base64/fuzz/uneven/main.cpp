#include <library/cpp/string_utils/base64/base64.h>

#include <util/system/types.h>
#include <util/system/yassert.h>

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    const TStringBuf example{reinterpret_cast<const char*>(data), size};
    Y_UNUSED(Base64DecodeUneven(example));
    return 0;
}
