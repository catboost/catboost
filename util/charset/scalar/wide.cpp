#include <util/charset/wide.h>
#include <util/system/types.h>

namespace NDetail {
    void UTF8ToWideImplSSE41(const unsigned char*& /*cur*/, const unsigned char* /*last*/, wchar16*& /*dest*/) noexcept {
    }
}
