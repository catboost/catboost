#pragma once

#include <util/system/defaults.h>
#include <util/system/unaligned_mem.h>

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
