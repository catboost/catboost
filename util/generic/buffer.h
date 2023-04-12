#pragma once

#include "utility.h"

#include <util/generic/fwd.h>
#include <util/system/align.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

#include <cstring>

class TBuffer {
public:
    using TIterator = char*;
    using TConstIterator = const char*;

    TBuffer(size_t len = 0);
    TBuffer(const char* buf, size_t len);

    TBuffer(const TBuffer& b)
        : Data_(nullptr)
        , Len_(0)
        , Pos_(0)
    {
        *this = b;
    }

    TBuffer(TBuffer&& b) noexcept;

    TBuffer& operator=(TBuffer&& b) noexcept;

    TBuffer& operator=(const TBuffer& b) {
        if (this != &b) {
            Assign(b.Data(), b.Size());
        }
        return *this;
    }

    ~TBuffer();

    inline void Clear() noexcept {
        Pos_ = 0;
    }

    inline void EraseBack(size_t n) noexcept {
        Y_ASSERT(n <= Pos_);
        Pos_ -= n;
    }

    Y_REINITIALIZES_OBJECT inline void Reset() noexcept {
        TBuffer().Swap(*this);
    }

    inline void Assign(const char* data, size_t len) {
        Clear();
        Append(data, len);
    }

    inline void Assign(const char* b, const char* e) {
        Assign(b, e - b);
    }

    inline char* Data() noexcept {
        return Data_;
    }

    inline const char* Data() const noexcept {
        return Data_;
    }

    inline char* Pos() noexcept {
        return Data_ + Pos_;
    }

    inline const char* Pos() const noexcept {
        return Data_ + Pos_;
    }

    /// Used space in bytes (do not mix with Capacity!)
    inline size_t Size() const noexcept {
        return Pos_;
    }

    Y_PURE_FUNCTION inline bool Empty() const noexcept {
        return !Size();
    }

    inline explicit operator bool() const noexcept {
        return Size();
    }

    inline size_t Avail() const noexcept {
        return Len_ - Pos_;
    }

    void Append(const char* buf, size_t len);

    inline void Append(const char* b, const char* e) {
        Append(b, e - b);
    }

    inline void Append(char ch) {
        if (Len_ == Pos_) {
            Reserve(Len_ + 1);
        }

        *(Data() + Pos_++) = ch;
    }

    void Fill(char ch, size_t len);

    // Method is useful when first messages from buffer are processed, and
    // the last message in buffer is incomplete, so we need to move partial
    // message to the begin of the buffer and continue filling the buffer
    // from the network.
    inline void Chop(size_t pos, size_t count) {
        const auto end = pos + count;
        Y_ASSERT(end <= Pos_);

        if (count == 0) {
            return;
        } else if (count == Pos_) {
            Pos_ = 0;
        } else {
            memmove(Data_ + pos, Data_ + end, Pos_ - end);
            Pos_ -= count;
        }
    }

    inline void ChopHead(size_t count) {
        Chop(0U, count);
    }

    inline void Proceed(size_t pos) {
        //Y_ASSERT(pos <= Len_); // see discussion in REVIEW:29021
        Resize(pos);
    }

    inline void Advance(size_t len) {
        Resize(Pos_ + len);
    }

    inline void Reserve(size_t len) {
        if (len > Len_) {
            DoReserve(len);
        }
    }

    inline void ReserveExactNeverCallMeInSaneCode(size_t len) {
        if (len > Len_) {
            Realloc(len);
        }
    }

    inline void ShrinkToFit() {
        if (Pos_ < Len_) {
            Realloc(Pos_);
        }
    }

    inline void Resize(size_t len) {
        Reserve(len);
        Pos_ = len;
    }

    // Method works like Resize, but allocates exact specified number of bytes
    // rather than rounded up to next power of 2
    // Use with care
    inline void ResizeExactNeverCallMeInSaneCode(size_t len) {
        ReserveExactNeverCallMeInSaneCode(len);
        Pos_ = len;
    }

    inline size_t Capacity() const noexcept {
        return Len_;
    }

    inline void AlignUp(size_t align, char fillChar = '\0') {
        size_t diff = ::AlignUp(Pos_, align) - Pos_;
        while (diff-- > 0) {
            Append(fillChar);
        }
    }

    inline char* data() noexcept {
        return Data();
    }

    inline const char* data() const noexcept {
        return Data();
    }

    inline size_t size() const noexcept {
        return Size();
    }

    inline void Swap(TBuffer& r) noexcept {
        DoSwap(Data_, r.Data_);
        DoSwap(Len_, r.Len_);
        DoSwap(Pos_, r.Pos_);
    }

    /*
     * after this call buffer becomes empty
     */
    void AsString(TString& s);

    /*
     * iterator-like interface
     */
    inline TIterator Begin() noexcept {
        return Data();
    }

    inline TIterator End() noexcept {
        return Begin() + Size();
    }

    inline TConstIterator Begin() const noexcept {
        return Data();
    }

    inline TConstIterator End() const noexcept {
        return Begin() + Size();
    }

    bool operator==(const TBuffer& other) const noexcept {
        if (Empty()) {
            return other.Empty();
        }
        return Size() == other.Size() && 0 == std::memcmp(Data(), other.Data(), Size());
    }

    bool operator!=(const TBuffer& other) const noexcept {
        return !(*this == other);
    }

private:
    void DoReserve(size_t len);
    void Realloc(size_t len);

private:
    char* Data_;
    size_t Len_;
    size_t Pos_;
};
