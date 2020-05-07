#include "md5.h"

#include <contrib/libs/nayuki_md5/md5.h>

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/stream/input.h>
#include <util/stream/file.h>
#include <library/cpp/string_utils/base64/base64.h>
#include <util/string/hex.h>

#include <cstring>
#include <cstdlib>

namespace {
    struct TMd5Stream: public IOutputStream {
        inline TMd5Stream(MD5* md5)
            : M_(md5)
        {
        }

        void DoWrite(const void* buf, size_t len) override {
            M_->Update(buf, len);
        }

        MD5* M_;
    };
}

char* MD5::File(const char* filename, char* buf) {
    try {
        TUnbufferedFileInput fi(filename);

        return Stream(&fi, buf);
    } catch (...) {
    }

    return nullptr;
}

TString MD5::File(const TString& filename) {
    char buf[33] = {0}; // 32 characters and \0
    return MD5::File(filename.data(), buf);
}

char* MD5::Data(const void* data, size_t len, char* buf) {
    MD5 md5;
    md5.Update(data, len);
    return md5.End(buf);
}

char* MD5::Stream(IInputStream* in, char* buf) {
    return MD5().Update(in).End(buf);
}

MD5& MD5::Update(IInputStream* in) {
    TMd5Stream md5(this);

    TransferData(in, &md5);

    return *this;
}

static inline void MD5Transform(ui32 state[4], const unsigned char block[64]) {
    return md5_compress((uint32_t*)state, (const ui8*)block);
}

/*
 * Encodes input (ui32) into output (unsigned char). Assumes len is
 * a multiple of 4.
 */

static void Encode(unsigned char* output, ui32* input, unsigned int len) {
    unsigned int i, j;

    for (i = 0, j = 0; j < len; i++, j += 4) {
        output[j] = (unsigned char)(input[i] & 0xff);
        output[j + 1] = (unsigned char)((input[i] >> 8) & 0xff);
        output[j + 2] = (unsigned char)((input[i] >> 16) & 0xff);
        output[j + 3] = (unsigned char)((input[i] >> 24) & 0xff);
    }
}

static unsigned char PADDING[64] = {
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* MD5 initialization. Begins an MD5 operation, writing a new context. */

void MD5::Init() {
    Count[0] = Count[1] = 0;
    /* Load magic initialization constants.  */
    State[0] = 0x67452301;
    State[1] = 0xefcdab89;
    State[2] = 0x98badcfe;
    State[3] = 0x10325476;
}

/*
 * MD5 block update operation. Continues an MD5 message-digest
 * operation, processing another message block, and updating the
 * context.
 */

void MD5::UpdatePart(const void* inputPtr, unsigned int inputLen) {
    const unsigned char* input = (const unsigned char*)inputPtr;
    unsigned int i, index, partLen;
    /* Compute number of bytes mod 64 */
    index = (unsigned int)((Count[0] >> 3) & 0x3F);
    /* Update number of bits */
    if ((Count[0] += ((ui32)inputLen << 3)) < ((ui32)inputLen << 3))
        Count[1]++;
    Count[1] += ((ui32)inputLen >> 29);
    partLen = 64 - index;
    /* Transform as many times as possible. */
    if (inputLen >= partLen) {
        memcpy((void*)&Buffer[index], (const void*)input, partLen);
        MD5Transform(State, Buffer);
        for (i = partLen; i + 63 < inputLen; i += 64)
            MD5Transform(State, &input[i]);
        index = 0;
    } else
        i = 0;
    /* Buffer remaining input */
    memcpy((void*)&Buffer[index], (const void*)&input[i], inputLen - i);
}

/*
 * MD5 padding. Adds padding followed by original length.
 */

void MD5::Pad() {
    unsigned char bits[8];
    unsigned int index, padLen;
    /* Save number of bits */
    Encode(bits, Count, 8);
    /* Pad out to 56 mod 64. */
    index = (unsigned int)((Count[0] >> 3) & 0x3f);
    padLen = (index < 56) ? (56 - index) : (120 - index);
    Update(PADDING, padLen);
    /* Append length (before padding) */
    Update(bits, 8);
}

/*
 * MD5 finalization. Ends an MD5 message-digest operation, writing the
 * the message digest and zeroizing the context.
 */

unsigned char* MD5::Final(unsigned char digest[16]) {
    /* Do padding. */
    Pad();
    /* Store state in digest */
    Encode(digest, State, 16);
    /* Zeroize sensitive information. */
    memset((void*)this, 0, sizeof(*this));

    return digest;
}

char* MD5::End(char* buf) {
    unsigned char digest[16];
    static const char hex[] = "0123456789abcdef";
    if (!buf)
        buf = (char*)malloc(33);
    if (!buf)
        return nullptr;
    Final(digest);
    int i = 0;
    for (; i < 16; i++) {
        buf[i + i] = hex[digest[i] >> 4];
        buf[i + i + 1] = hex[digest[i] & 0x0f];
    }
    buf[i + i] = '\0';
    return buf;
}

char* MD5::End_b64(char* buf) {
    unsigned char digest[16];
    if (!buf)
        buf = (char*)malloc(25);
    if (!buf)
        return nullptr;
    Final(digest);
    Base64Encode(buf, digest, 16);
    buf[24] = '\0';
    return buf;
}

ui64 MD5::EndHalfMix() {
    unsigned char digest[16];
    Final(digest);
    ui64 res = 0;
    for (int i = 3; i >= 0; i--) {
        res |= (ui64)(digest[0 + i] ^ digest[8 + i]) << ((3 - i) << 3);
        res |= (ui64)(digest[4 + i] ^ digest[12 + i]) << ((7 - i) << 3);
    }
    return res;
}

TString MD5::Calc(const TStringBuf& data) {
    TString result;
    result.resize(32);

    Data((const unsigned char*)data.data(), data.size(), result.begin());

    return result;
}

TString MD5::CalcRaw(const TStringBuf& data) {
    TString result;
    result.resize(16);
    MD5 md5;
    md5.Update(data.data(), data.size());
    md5.Final(reinterpret_cast<unsigned char*>(result.begin()));
    return result;
}

ui64 MD5::CalcHalfMix(const char* data, size_t len) {
    MD5 md5;
    md5.Update(data, len);
    return md5.EndHalfMix();
}

ui64 MD5::CalcHalfMix(const TStringBuf& data) {
    return CalcHalfMix(data.data(), data.size());
}

bool MD5::IsMD5(const TStringBuf& data) {
    if (data.size() != 32) {
        return false;
    }
    for (const char *p = data.data(), *e = data.data() + data.size(); p != e; ++p) {
        if (Char2DigitTable[(unsigned char)*p] == '\xff') {
            return false;
        }
    }
    return true;
}
