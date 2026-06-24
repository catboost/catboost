#pragma once

#include <util/generic/ptr.h>
#include <util/stream/input.h>
#include <util/stream/output.h>

/**
 * @addtogroup Streams_Archs
 * @{
 */

// @brief Stream to compress into zstd archive
class TZstdCompress: public IOutputStream {
public:
    /**
      @param slave stream to write compressed data to
      @param quality, higher quality - slower but better compression.
             0 is default compression (see constant ZSTD_CLEVEL_DEFAULT(3))
             max compression is  ZSTD_MAX_CLEVEL (22)
    */
    explicit TZstdCompress(IOutputStream* slave, int quality = 0);
    ~TZstdCompress() override;
private:
    void DoWrite(const void* buffer, size_t size) override;
    void DoFlush() override;
    void DoFinish() override;

public:
    class TImpl;
    THolder<TImpl> Impl_;
};

////////////////////////////////////////////////////////////////////////////////

// @brief Buffered stream to decompress from zstd archive
class TZstdDecompress: public IInputStream {
public:
    /**
      @param slave stream to read compressed data from
      @param bufferSize approximate size of buffer compressed data is read in
    */
    explicit TZstdDecompress(IInputStream* slave, size_t bufferSize = 8 * 1024);
    ~TZstdDecompress() override;

private:
    size_t DoRead(void* buffer, size_t size) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/** @} */
