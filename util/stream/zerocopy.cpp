#include "zerocopy.h"
#include "output.h"

IZeroCopyInput::~IZeroCopyInput() = default;

size_t IZeroCopyInput::DoRead(void* buf, size_t len) {
    const void* ptr;
    size_t result = DoNext(&ptr, len);

    if (result) {
        memcpy(buf, ptr, result);
    }

    return result;
}

ui64 IZeroCopyInput::DoReadAll(IOutputStream& out) {
    ui64 result = 0;
    const void* ptr;

    while (size_t len = Next(&ptr)) {
        out.Write(ptr, len);
        result += len;
    }

    return result;
}

size_t IZeroCopyInput::DoSkip(size_t len) {
    const void* ptr;

    return DoNext(&ptr, len);
}

IZeroCopyInputFastReadTo::~IZeroCopyInputFastReadTo() = default;

size_t IZeroCopyInputFastReadTo::DoReadTo(TString& st, char ch) {
    const char* ptr;
    size_t len = Next(&ptr);
    if (!len) {
        return 0;
    }
    size_t result = 0;
    st.clear();
    do {
        if (const char* pos = (const char*)memchr(ptr, ch, len)) {
            size_t bytesRead = (pos - ptr) + 1;
            if (bytesRead > 1) {
                st.append(ptr, pos);
            }
            Undo(len - bytesRead);
            result += bytesRead;
            return result;
        } else {
            result += len;
            st.append(ptr, len);
        }
        len = Next(&ptr);
    } while (len);
    return result;
}
