#pragma once

#include "zerocopy.h"
#include "zerocopy_output.h"

#include <util/generic/strbuf.h>

/**
 * @addtogroup Streams_Memory
 * @{
 */

/**
 * Input stream that reads data from a memory block.
 */
class TMemoryInput: public IZeroCopyInputFastReadTo {
public:
    TMemoryInput() noexcept;

    /**
     * Constructs a stream that reads from the provided memory block. It's up
     * to the user to make sure that the memory block doesn't get freed while
     * this stream is in use.
     *
     * @param buf                       Memory block to use.
     * @param len                       Size of the memory block.
     */
    TMemoryInput(const void* buf Y_LIFETIME_BOUND, size_t len) noexcept;
    TMemoryInput(TString&&) = delete;
    explicit TMemoryInput(const TStringBuf buf Y_LIFETIME_BOUND) noexcept;
    explicit TMemoryInput(const char* zstr Y_LIFETIME_BOUND)
        : TMemoryInput(TStringBuf(zstr))
    {
    }

    ~TMemoryInput() override;

    TMemoryInput(const TMemoryInput& other) noexcept
        : IZeroCopyInputFastReadTo()
        , Buf_(other.Buf_)
        , Len_(other.Len_)
    {
    }

    TMemoryInput& operator=(const TMemoryInput& other) noexcept {
        if (this != &other) {
            Buf_ = other.Buf_;
            Len_ = other.Len_;
        }

        return *this;
    }

    TMemoryInput(TMemoryInput&&) noexcept = default;
    TMemoryInput& operator=(TMemoryInput&&) noexcept = default;

    /**
     * Initializes this stream with a new memory block. It's up to the
     * user to make sure that the memory block doesn't get freed while this
     * stream is in use.
     *
     * @param buf                       New memory block to use.
     * @param len                       Size of the new memory block.
     */
    void Reset(const void* buf, size_t len) noexcept {
        Buf_ = (const char*)buf;
        Len_ = len;
    }

    /**
     * @returns                         Whether there is more data in the stream.
     */
    bool Exhausted() const noexcept {
        return !Avail();
    }

    /**
     * @returns                         Number of bytes available in the stream.
     */
    size_t Avail() const noexcept {
        return Len_;
    }

    /**
     * @returns                         Current read position in the memory block
     *                                  used by this stream.
     */
    const char* Buf() const noexcept {
        return Buf_;
    }

    /**
     * Initializes this stream with a next chunk extracted from the given zero
     * copy stream.
     *
     * @param stream                    Zero copy stream to initialize from.
     */
    void Fill(IZeroCopyInput* stream) {
        Len_ = stream->Next(&Buf_);
        if (!Len_) {
            Reset(nullptr, 0);
        }
    }

private:
    size_t DoNext(const void** ptr, size_t len) override;
    void DoUndo(size_t len) override;

private:
    const char* Buf_;
    size_t Len_;
};

/**
 * Output stream that writes data to a memory block.
 */
class TMemoryOutput: public IZeroCopyOutput {
public:
    /**
     * Constructs a stream that writes to the provided memory block. It's up
     * to the user to make sure that the memory block doesn't get freed while
     * this stream is in use.
     *
     * @param buf                       Memory block to use.
     * @param len                       Size of the memory block.
     */
    TMemoryOutput(void* buf, size_t len) noexcept
        : Buf_(static_cast<char*>(buf))
        , End_(Buf_ + len)
    {
    }
    ~TMemoryOutput() override;

    TMemoryOutput(TMemoryOutput&&) noexcept = default;
    TMemoryOutput& operator=(TMemoryOutput&&) noexcept = default;

    /**
     * Initializes this stream with a new memory block. It's up to the
     * user to make sure that the memory block doesn't get freed while this
     * stream is in use.
     *
     * @param buf                       New memory block to use.
     * @param len                       Size of the new memory block.
     */
    inline void Reset(void* buf, size_t len) noexcept {
        Buf_ = static_cast<char*>(buf);
        End_ = Buf_ + len;
    }

    /**
     * @returns                         Whether there is more space in the
     *                                  stream for writing.
     */
    inline bool Exhausted() const noexcept {
        return !Avail();
    }

    /**
     * @returns                         Number of bytes available for writing
     *                                  in the stream.
     */
    inline size_t Avail() const noexcept {
        return End_ - Buf_;
    }

    /**
     * @returns                         Current write position in the memory block
     *                                  used by this stream.
     */
    inline char* Buf() const noexcept {
        return Buf_;
    }

    /**
     * @returns                         Pointer to the end of the memory block
     *                                  used by this stream.
     */
    char* End() const {
        return End_;
    }

private:
    size_t DoNext(void** ptr) override;
    void DoUndo(size_t len) override;
    void DoWrite(const void* buf, size_t len) override;
    void DoWriteC(char c) override;

protected:
    char* Buf_;
    char* End_;
};

/**
 * Memory output stream that supports changing the position of the
 * write pointer.
 *
 * @see TMemoryOutput
 */
class TMemoryWriteBuffer: public TMemoryOutput {
public:
    TMemoryWriteBuffer(void* buf, size_t len)
        : TMemoryOutput(buf, len)
        , Beg_(Buf_)
    {
    }

    void Reset(void* buf, size_t len) {
        TMemoryOutput::Reset(buf, len);
        Beg_ = Buf_;
    }

    size_t Len() const {
        return Buf() - Beg();
    }

    size_t Empty() const {
        return Buf() == Beg();
    }

    /**
     * @returns                         Data that has been written into this
     *                                  stream as a string.
     */
    TStringBuf Str() const {
        return TStringBuf(Beg(), Buf());
    }

    char* Beg() const {
        return Beg_;
    }

    /**
     * @param ptr                       New write position for this stream.
     *                                  Must be inside the memory block that
     *                                  this stream uses.
     */
    void SetPos(char* ptr) {
        Y_ASSERT(Beg_ <= ptr);
        SetPosImpl(ptr);
    }

    /**
     * @param pos                       New write position for this stream,
     *                                  relative to the beginning of the memory
     *                                  block that this stream uses.
     */
    void SetPos(size_t pos) {
        SetPosImpl(Beg_ + pos);
    }

protected:
    void SetPosImpl(char* ptr) {
        Y_ASSERT(End_ >= ptr);
        Buf_ = ptr;
    }

protected:
    char* Beg_;
};

/** @} */
