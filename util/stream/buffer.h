#pragma once

#include "zerocopy.h"
#include "zerocopy_output.h"

#include <util/generic/ptr.h>

class TBuffer;

/**
 * @addtogroup Streams_Buffers
 * @{
 */

/**
 * Output stream that writes into a `TBuffer`.
 */
class TBufferOutput: public IZeroCopyOutput {
public:
    class TImpl;

    /**
     * Constructs a stream that writes into an internal buffer.
     *
     * @param buflen                    Initial size of the internal buffer.
     */
    TBufferOutput(size_t buflen = 1024);

    /**
     * Constructs a stream that writes into the provided buffer. It's up to the
     * user to make sure that the buffer doesn't get destroyed while this stream
     * is in use.
     *
     * @param buffer                    Buffer to write into.
     */
    TBufferOutput(TBuffer& buffer);

    TBufferOutput(TBufferOutput&&) noexcept;
    TBufferOutput& operator=(TBufferOutput&&) noexcept;

    ~TBufferOutput() override;

    /**
     * @returns                         Buffer that this stream writes into.
     */
    TBuffer& Buffer() const noexcept;

private:
    size_t DoNext(void** ptr) override;
    void DoUndo(size_t len) override;
    void DoWrite(const void* buf, size_t len) override;
    void DoWriteC(char c) override;

private:
    THolder<TImpl> Impl_;
};

/**
 * Input stream that reads from an external `TBuffer`.
 */
class TBufferInput: public IZeroCopyInputFastReadTo {
public:
    /**
     * Constructs a stream that reads from an external buffer. It's up to the
     * user to make sure that the buffer doesn't get destroyed before this
     * stream.
     *
     * @param buffer                    External buffer to read from.
     */
    TBufferInput(const TBuffer& buffer);

    ~TBufferInput() override;

    const TBuffer& Buffer() const noexcept;

    void Rewind() noexcept;

protected:
    size_t DoNext(const void** ptr, size_t len) override;
    void DoUndo(size_t len) override;

private:
    const TBuffer& Buf_;
    size_t Readed_;
};

/**
 * Input/output stream that works with a `TBuffer`.
 */
class TBufferStream: public TBufferOutput, public TBufferInput {
public:
    /**
     * Constructs a stream that works with an internal buffer.
     *
     * @param buflen                    Initial size of the internal buffer.
     */
    inline TBufferStream(size_t buflen = 1024)
        : TBufferOutput(buflen)
        , TBufferInput(TBufferOutput::Buffer())
    {
    }

    /**
     * Constructs a stream that works with the provided buffer.
     *
     * @param buffer                    Buffer to work with.
     */
    inline TBufferStream(TBuffer& buffer)
        : TBufferOutput(buffer)
        , TBufferInput(TBufferOutput::Buffer())
    {
    }

    ~TBufferStream() override = default;

    using TBufferOutput::Buffer;
};

/** @} */
