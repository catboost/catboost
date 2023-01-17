#pragma once

#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/generic/ptr.h>

/**
 * @addtogroup Streams_Archs
 * @{
 */

/**
 * Snappy compressing stream.
 *
 * @see http://code.google.com/p/snappy/
 */
class TSnappyCompress: public IOutputStream {
public:
    TSnappyCompress(IOutputStream* slave, ui16 maxBlockSize = 1 << 15);
    ~TSnappyCompress() override;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/**
 * Snappy decompressing stream.
 *
 * @see http://code.google.com/p/snappy/
 */
class TSnappyDecompress: public IInputStream {
public:
    TSnappyDecompress(IInputStream* slave);
    ~TSnappyDecompress() override;

private:
    size_t DoRead(void* buf, size_t len) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/** @} */
