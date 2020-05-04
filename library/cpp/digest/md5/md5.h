#pragma once

#include <util/system/defaults.h>
#include <util/generic/string.h>
#include <util/generic/strbuf.h>

class IInputStream;

class MD5 {
public:
    MD5() {
        Init();
    }

    void Init();

    inline MD5& Update(const void* data, size_t len) {
        const char* buf = (const char*)data;

        while (len) {
            // NOTE: we don't want buffSz to be near Max<unsigned int>()
            // because otherwise integer overflow might happen in UpdatePart
            const unsigned int buffSz = Min(size_t(Max<unsigned int>() / 2), len);

            UpdatePart(buf, buffSz);
            buf += buffSz;
            len -= buffSz;
        }
        return *this;
    }

    inline MD5& Update(const TStringBuf& data) {
        return Update(data.data(), data.size());
    }

    void Pad();
    unsigned char* Final(unsigned char[16]);

    // buf must be char[33];
    char* End(char* buf);

    // buf must be char[25];
    char* End_b64(char* buf);

    // 8-byte xor-based mix
    ui64 EndHalfMix();

    MD5& Update(IInputStream* in);

    // return hex-encoded data
    static char* File(const char* filename, char* buf);
    static TString File(const TString& filename);
    static char* Data(const void* data, size_t len, char* buf);
    static char* Stream(IInputStream* in, char* buf);

    static TString Calc(const TStringBuf& data);    // 32-byte hex-encoded
    static TString CalcRaw(const TStringBuf& data); // 16-byte raw

    static ui64 CalcHalfMix(const TStringBuf& data);
    static ui64 CalcHalfMix(const char* data, size_t len);

    static bool IsMD5(const TStringBuf& data);

private:
    void UpdatePart(const void* data, unsigned int len);

private:
    ui32 State[4];            /* state (ABCD) */
    ui32 Count[2];            /* number of bits, modulo 2^64 (lsb first) */
    unsigned char Buffer[64]; /* input buffer */
};
