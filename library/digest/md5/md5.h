#pragma once

#include <util/system/defaults.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>

class TInputStream;

class MD5 {
public:
    MD5() {
        Init();
    }

    void Init();

    inline MD5& Update(const void* data, size_t len) {
        const char* buf = (const char*)data;

        while (len) {
            const unsigned int buffSz = (unsigned int)Min((size_t)Max<unsigned int>(), len);

            UpdatePart(buf, buffSz);
            buf += buffSz;
            len -= buffSz;
        }
        return *this;
    }

    inline MD5& Update(const TStringBuf& data) {
        return Update(~data, +data);
    }

    void UpdatePart(const void* data, unsigned int len);
    void Pad();
    unsigned char* Final(unsigned char[16]);

    // buf must be char[33];
    char* End(char* buf);

    // buf must be char[25];
    char* End_b64(char* buf);

    MD5& Update(TInputStream* in);

    // return hex-encoded data
    static char* File(const char* filename, char* buf);
    static TString File(const TString& filename);
    static char* Data(const void* data, size_t len, char* buf);
    static char* Stream(TInputStream* in, char* buf);

    static TString Calc(const TStringBuf& data);    // 32-byte hex-encoded
    static TString CalcRaw(const TStringBuf& data); // 16-byte raw

    static bool IsMD5(const TStringBuf& data);

private:
    ui32 State[4];            /* state (ABCD) */
    ui32 Count[2];            /* number of bits, modulo 2^64 (lsb first) */
    unsigned char Buffer[64]; /* input buffer */
};
