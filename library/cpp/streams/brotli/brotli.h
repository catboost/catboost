#pragma once

#include <util/generic/ptr.h>
#include <util/stream/input.h>
#include <util/stream/output.h>

/**
 * @addtogroup Streams_Archs
 * @{
 */

class TBrotliCompress: public IOutputStream {
public:
    static constexpr int BEST_QUALITY = 11;

    /**
      @param slave stream to write compressed data to
      @param quality the higher the quality, the slower and better the compression. Range is 0 to 11.
    */
    explicit TBrotliCompress(IOutputStream* slave, int quality = BEST_QUALITY);
    ~TBrotliCompress() override;

private:
    void DoWrite(const void* buffer, size_t size) override;
    void DoFlush() override;
    void DoFinish() override;

public:
    class TImpl;
    THolder<TImpl> Impl_;
};

////////////////////////////////////////////////////////////////////////////////

class TBrotliDecompress: public IInputStream {
public:
    /**
      @param slave stream to read compressed data from
      @param bufferSize approximate size of buffer compressed data is read in
    */
    explicit TBrotliDecompress(IInputStream* slave, size_t bufferSize = 8 * 1024);
    ~TBrotliDecompress() override;

private:
    size_t DoRead(void* buffer, size_t size) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/** @} */
