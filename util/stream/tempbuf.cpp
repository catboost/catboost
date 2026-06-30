#include "tempbuf.h"

namespace {
    static inline size_t Next(size_t size) noexcept {
        return size * 2;
    }
} // namespace

void TTempBufOutput::DoWrite(const void* data, size_t len) {
    if (Y_LIKELY(len <= Left())) {
        Append(data, len);
    } else {
        const size_t filled = Filled();

        TTempBuf buf(Next(filled + len));

        buf.Append(Data(), filled);
        buf.Append(data, len);

        static_cast<TTempBuf&>(*this) = buf;
    }
}
