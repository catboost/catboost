#pragma once

#include "const.h"
#include "dictionary.h"

#include <util/generic/ptr.h>
#include <util/stream/input.h>
#include <util/stream/output.h>

/**
 * @addtogroup Streams_Archs
 * @{
 */

class TBrotliCompress: public IOutputStream {
public:
    /**
      @param slave stream to write compressed data to
      @param quality the higher the quality, the slower and better the compression. Range is 0 to 11.
      @param dictionary custom brotli dictionary
      @param offset number of bytes already processed by a different encoder instance
    */
    explicit TBrotliCompress(IOutputStream* slave,
                             int quality = NBrotli::BEST_BROTLI_QUALITY,
                             const TBrotliDictionary* dictionary = nullptr,
                             size_t offset = 0);
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
      @param dictionary custom brotli dictionary
    */
    explicit TBrotliDecompress(IInputStream* slave,
                               size_t bufferSize = NBrotli::DEFAULT_BROTLI_BUFFER_SIZE,
                               const TBrotliDictionary* dictionary = nullptr);
    ~TBrotliDecompress() override;

private:
    size_t DoRead(void* buffer, size_t size) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/** @} */
