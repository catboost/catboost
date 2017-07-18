#include "zerocopy.h"
#include "output.h"

TZeroCopyInput::~TZeroCopyInput() = default;

size_t TZeroCopyInput::DoRead(void* buf, size_t len) {
    const void* ptr;
    size_t result = DoNext(&ptr, len);

    if (result) {
        memcpy(buf, ptr, result);
    }

    return result;
}

ui64 TZeroCopyInput::DoReadAll(TOutputStream& out) {
    ui64 result = 0;
    const void* ptr;

    while (size_t len = Next(&ptr)) {
        out.Write(ptr, len);
        result += len;
    }

    return result;
}

size_t TZeroCopyInput::DoSkip(size_t len) {
    const void* ptr;

    return DoNext(&ptr, len);
}

TZeroCopyInputFastReadTo::~TZeroCopyInputFastReadTo() = default;

size_t TZeroCopyInputFastReadTo::DoReadTo(TString& st, char ch) {
    const char* ptr;
    size_t len = Next(&ptr);
    if (!len) {
        return 0;
    }
    size_t result = 0;
    st.clear();
    do {
        if (const char* pos = (const char*)memchr(ptr, ch, len)) {
            size_t readed = (pos - ptr) + 1;
            if (readed > 1) {
                st.append(ptr, pos);
            }
            Undo(len - readed);
            result += readed;
            return result;
        } else {
            result += len;
            st.append(ptr, len);
        }
    } while (len = Next(&ptr));
    return result;
}
