#pragma once

#include <util/stream/output.h>
#include <util/stream/input.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>

#include <library/cpp/streams/lz/common/error.h>

#include <library/cpp/streams/lz/lz4/lz4.h>
#include <library/cpp/streams/lz/snappy/snappy.h>

/**
 * @file
 *
 * All lz compressors compress blocks. `Write` method splits input data into
 * blocks, compresses each block and then writes each compressed block to the
 * underlying output stream. Thus compression classes are not buffered.
 * MaxBlockSize parameter specified max allowed block size.
 *
 * See http://altdevblogaday.com/2011/04/22/survey-of-fast-compression-algorithms-part-1/
 * for some comparisons.
 */

/**
 * @addtogroup Streams_Archs
 * @{
 */

#ifndef OPENSOURCE

/**
 * MiniLZO compressing stream.
 */
class TLzoCompress: public IOutputStream {
public:
    TLzoCompress(IOutputStream* slave, ui16 maxBlockSize = 1 << 15);
    ~TLzoCompress() override;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/**
 * MiniLZO decompressing stream.
 */
class TLzoDecompress: public IInputStream {
public:
    TLzoDecompress(IInputStream* slave);
    ~TLzoDecompress() override;

private:
    size_t DoRead(void* buf, size_t len) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

#endif

/**
 * FastLZ compressing stream.
 */
class TLzfCompress: public IOutputStream {
public:
    TLzfCompress(IOutputStream* slave, ui16 maxBlockSize = 1 << 15);
    ~TLzfCompress() override;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/**
 * FastLZ decompressing stream.
 */
class TLzfDecompress: public IInputStream {
public:
    TLzfDecompress(IInputStream* slave);
    ~TLzfDecompress() override;

private:
    size_t DoRead(void* buf, size_t len) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

#ifndef OPENSOURCE

/**
 * QuickLZ compressing stream.
 */
class TLzqCompress: public IOutputStream {
public:
    enum EVersion {
        V_1_31 = 0,
        V_1_40 = 1,
        V_1_51 = 2
    };

    /*
     * streaming mode - actually, backlog size
     */
    enum EMode {
        M_0 = 0,
        M_100000 = 1,
        M_1000000 = 2
    };

    TLzqCompress(IOutputStream* slave, ui16 maxBlockSize = 1 << 15,
                 EVersion ver = V_1_31,
                 unsigned level = 0,
                 EMode mode = M_0);
    ~TLzqCompress() override;

private:
    void DoWrite(const void* buf, size_t len) override;
    void DoFlush() override;
    void DoFinish() override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

/**
 * QuickLZ decompressing stream.
 */
class TLzqDecompress: public IInputStream {
public:
    TLzqDecompress(IInputStream* slave);
    ~TLzqDecompress() override;

private:
    size_t DoRead(void* buf, size_t len) override;

private:
    class TImpl;
    THolder<TImpl> Impl_;
};

#endif

/** @} */

/**
 * Reads a compression signature from the provided input stream and returns a
 * corresponding decompressing stream.
 *
 * Note that returned stream doesn't own the provided input stream, thus it's
 * up to the user to free them both.
 *
 * @param input                         Stream to decompress.
 * @return                              Decompressing proxy input stream.
 */
TAutoPtr<IInputStream> OpenLzDecompressor(IInputStream* input);
TAutoPtr<IInputStream> TryOpenLzDecompressor(IInputStream* input);
TAutoPtr<IInputStream> TryOpenLzDecompressor(const TStringBuf& signature, IInputStream* input);

TAutoPtr<IInputStream> OpenOwnedLzDecompressor(TAutoPtr<IInputStream> input);
TAutoPtr<IInputStream> TryOpenOwnedLzDecompressor(TAutoPtr<IInputStream> input);
TAutoPtr<IInputStream> TryOpenOwnedLzDecompressor(const TStringBuf& signature, TAutoPtr<IInputStream> input);
