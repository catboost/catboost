#pragma once

#include <util/system/yassert.h>
#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <string.h>

struct IBinaryStream {
    virtual ~IBinaryStream() = default;

    inline i64 Write(const void* userBuffer, i64 size) {
        if (size <= Max<int>()) {
            return WriteImpl(userBuffer, static_cast<int>(size));
        } else {
            return LongWrite(userBuffer, size);
        }
    }

    inline i64 Read(void* userBuffer, i64 size) {
        if (size <= Max<int>()) {
            return ReadImpl(userBuffer, static_cast<int>(size));
        } else {
            return LongRead(userBuffer, size);
        }
    }

    virtual bool IsValid() const = 0;
    virtual bool IsFailed() const = 0;

private:
    virtual int WriteImpl(const void* userBuffer, int size) = 0;
    virtual int ReadImpl(void* userBuffer, int size) = 0;

    i64 LongRead(void* userBuffer, i64 size);
    i64 LongWrite(const void* userBuffer, i64 size);
};

template <int N_SIZE = 16384>
class TBufferedStream {
    char Buf[N_SIZE];
    i64 Pos, BufSize;
    IBinaryStream& Stream;
    bool bIsReading, bIsEof, bFailed;

    void ReadComplex(void* userBuffer, i64 size) {
        if (bIsEof) {
            memset(userBuffer, 0, size);
            return;
        }
        char* dst = (char*)userBuffer;
        i64 leftBytes = BufSize - Pos;
        memcpy(dst, Buf + Pos, leftBytes);
        dst += leftBytes;
        size -= leftBytes;
        Pos = BufSize = 0;
        if (size > N_SIZE) {
            i64 n = Stream.Read(dst, size);
            bFailed = Stream.IsFailed();
            if (n != size) {
                bIsEof = true;
                memset(dst + n, 0, size - n);
            }
        } else {
            BufSize = Stream.Read(Buf, N_SIZE);
            bFailed = Stream.IsFailed();
            if (BufSize == 0)
                bIsEof = true;
            Read(dst, size);
        }
    }

    void WriteComplex(const void* userBuffer, i64 size) {
        Flush();
        if (size >= N_SIZE) {
            Stream.Write(userBuffer, size);
            bFailed = Stream.IsFailed();
        } else
            Write(userBuffer, size);
    }

    void operator=(const TBufferedStream&) {
    }

public:
    TBufferedStream(bool bRead, IBinaryStream& stream)
        : Pos(0)
        , BufSize(0)
        , Stream(stream)
        , bIsReading(bRead)
        , bIsEof(false)
        , bFailed(false)
    {
    }
    ~TBufferedStream() {
        if (!bIsReading)
            Flush();
    }
    void Flush() {
        Y_ASSERT(!bIsReading);
        if (bIsReading)
            return;
        Stream.Write(Buf, Pos);
        bFailed = Stream.IsFailed();
        Pos = 0;
    }
    bool IsEof() const {
        return bIsEof;
    }
    inline void Read(void* userBuffer, i64 size) {
        Y_ASSERT(bIsReading);
        if (!bIsEof && size + Pos <= BufSize) {
            memcpy(userBuffer, Buf + Pos, size);
            Pos += size;
            return;
        }
        ReadComplex(userBuffer, size);
    }
    inline void Write(const void* userBuffer, i64 size) {
        Y_ASSERT(!bIsReading);
        if (Pos + size < N_SIZE) {
            memcpy(Buf + Pos, userBuffer, size);
            Pos += size;
            return;
        }
        WriteComplex(userBuffer, size);
    }
    bool IsValid() const {
        return Stream.IsValid();
    }
    bool IsFailed() const {
        return bFailed;
    }
};
