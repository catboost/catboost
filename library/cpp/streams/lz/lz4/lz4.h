#pragma once

#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/generic/ptr.h>

/**
 * @addtogroup Streams_Archs
 * @{
 */

/**
 * Lz4 compressing stream.
 *
 * @see http://code.google.com/p/lz4/
 */
class TLz4Compress: public IOutputStream {
public:
    TLz4Compress(IOutputStream* slave, ui16 maxBlockSize = 1 << 15);
    ~TLz4Compress() override;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/**
 * Lz4 decompressing stream.
 *
 * @see http://code.google.com/p/lz4/
 */
class TLz4Decompress: public IInputStream {
public:
    TLz4Decompress(IInputStream* slave);
    ~TLz4Decompress() override;

private:
    size_t DoRead(void* buf, size_t len) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/** @} */
