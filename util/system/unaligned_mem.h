#pragma once

#include "defaults.h"
#include "yassert.h"

#include <string.h>

// The following code used to have smart tricks assuming that unaligned reads and writes are OK on x86. This assumption
// is wrong because compiler may emit alignment-sensitive x86 instructions e.g. movaps. See IGNIETFERRO-735.

template <class T>
inline T ReadUnaligned(const void* from) noexcept {
    T ret;
    memcpy(&ret, from, sizeof(T));
    return ret;
}

template <class T>
inline void WriteUnaligned(void* to, const T& t) noexcept {
    memcpy(to, &t, sizeof(T));
}

template <class T, unsigned Align = sizeof(T)>
class TUnalignedMemoryIterator {
public:
    inline TUnalignedMemoryIterator(const void* buf, size_t len)
        : C_((const unsigned char*)buf)
        , E_(C_ + len)
        , L_(E_ - (len % Align))
    {
        Y_FAKE_READ(buf);
    }

    inline bool AtEnd() const noexcept {
        return C_ == L_;
    }

    inline T Cur() const noexcept {
        Y_ASSERT(C_ < L_);
        return ::ReadUnaligned<T>(C_);
    }

    inline T Next() noexcept {
        T ret(Cur());

        C_ += sizeof(T);

        return ret;
    }

    inline const unsigned char* Last() const noexcept {
        return C_;
    }

    inline size_t Left() const noexcept {
        return E_ - C_;
    }

private:
    const unsigned char* C_;
    const unsigned char* E_;
    const unsigned char* L_;
};
