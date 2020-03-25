#pragma once

#include <util/stream/input.h>
#include <util/stream/output.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>

#define BZIP_BUF_LEN (8 * 1024)
#define BZIP_COMPRESSION_LEVEL 6

/**
 * @addtogroup Streams_Archs
 * @{
 */

class TBZipException: public yexception {
};

class TBZipDecompressError: public TBZipException {
};

class TBZipCompressError: public TBZipException {
};

class TBZipDecompress: public IInputStream {
public:
    TBZipDecompress(IInputStream* input, size_t bufLen = BZIP_BUF_LEN);
    ~TBZipDecompress() override;

private:
    size_t DoRead(void* buf, size_t size) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

class TBZipCompress: public IOutputStream {
public:
    TBZipCompress(IOutputStream* out, size_t compressionLevel = BZIP_COMPRESSION_LEVEL, size_t bufLen = BZIP_BUF_LEN);
    ~TBZipCompress() override;

private:
    void DoWrite(const void* buf, size_t size) override;
    void DoFlush() override;
    void DoFinish() override;

public:
    class TImpl;
    THolder<TImpl> Impl_;
};

/** @} */
