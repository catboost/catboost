#include "md5.h"

#include <library/cpp/string_utils/base64/base64.h>

#include <util/stream/input.h>
#include <util/stream/file.h>
#include <util/string/hex.h>

#include <contrib/libs/nayuki_md5/md5.h>

namespace {

    constexpr size_t MD5_BLOCK_LENGTH = 64;
    constexpr size_t MD5_PADDING_SHIFT = 56;
    constexpr size_t MD5_HEX_DIGEST_LENGTH = 32;

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

    inline TArrayRef<const ui8> MakeUnsignedArrayRef(const void* data, const size_t size) {
        return MakeArrayRef(static_cast<const ui8*>(data), size);
    }

    inline TArrayRef<const ui8> MakeUnsignedArrayRef(const TArrayRef<const char>& data) {
        return MakeUnsignedArrayRef(data.data(), data.size());
    }
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
    TString buf;
    buf.ReserveAndResize(MD5_HEX_DIGEST_LENGTH);
    auto result = MD5::File(filename.data(), buf.begin());
    if (result == nullptr) {
        buf.clear();
    }
    return buf;
}

char* MD5::Data(const TArrayRef<const ui8>& data, char* buf) {
    return MD5().Update(data).End(buf);
}

char* MD5::Data(const void* data, size_t len, char* buf) {
    return Data(MakeUnsignedArrayRef(data, len), buf);
}

TString MD5::Data(const TArrayRef<const ui8>& data) {
    TString buf;
    buf.ReserveAndResize(MD5_HEX_DIGEST_LENGTH);
    Data(data, buf.begin());
    return buf;
}

TString MD5::Data(TStringBuf data) {
    return Data(MakeUnsignedArrayRef(data));
}

char* MD5::Stream(IInputStream* in, char* buf) {
    return MD5().Update(in).End(buf);
}

MD5& MD5::Update(IInputStream* in) {
    TMd5Stream md5(this);

    TransferData(in, &md5);

    return *this;
}

static const ui8 PADDING[MD5_BLOCK_LENGTH] = {
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* MD5 initialization. Begins an MD5 operation, writing a new context. */

void MD5::Init() {
    BufferSize = 0;
    StreamSize = 0;
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

void MD5::UpdatePart(TArrayRef<const ui8> data) {
    /* Count input bytes */
    StreamSize += data.size();
    if (BufferSize > 0) {
        /* Filling remaining buffer */
        const ui8 freeBufferSize = MD5_BLOCK_LENGTH - BufferSize;
        const ui8 partLen = data.size() >= freeBufferSize ? freeBufferSize : data.size();
        memcpy(&Buffer[BufferSize], data.data(), partLen);
        BufferSize += partLen;
        data = data.Slice(partLen);
        if (BufferSize == MD5_BLOCK_LENGTH) {
            /* Buffer is full and ready for hashing */
            md5_compress(State, Buffer);
            BufferSize = 0;
        }
    }
    /* Processing input by chanks */
    while (data.size() >= MD5_BLOCK_LENGTH) {
        md5_compress(State, data.data());
        data = data.Slice(MD5_BLOCK_LENGTH);
    }
    /* Save remaining input in buffer */
    memcpy(Buffer, data.data(), data.size());
    BufferSize += data.size();
}

/*
 * MD5 padding. Adds padding followed by original length.
 */

void MD5::Pad() {
    size_t streamSize = StreamSize;

    const size_t paddingSize = (MD5_PADDING_SHIFT > BufferSize) ? (MD5_PADDING_SHIFT - BufferSize) : (MD5_PADDING_SHIFT + MD5_BLOCK_LENGTH - BufferSize);
    Update(PADDING, paddingSize);

    // Size of stream in bits
    // If size greater than 2^64 - 1 only lower 64 bits used
    streamSize <<= 3;
    for (int i = 0; i < 8; ++i, streamSize >>= 8) {
        // Storing in reverse order
        Buffer[MD5_PADDING_SHIFT + i] = static_cast<ui8>(streamSize & 0xFFU);
    }
    md5_compress(State, Buffer);
}

/*
 * MD5 finalization. Ends an MD5 message-digest operation, writing the
 * the message digest and zeroizing the context.
 */

ui8* MD5::Final(ui8 digest[16]) {
    /* Do padding. */
    Pad();
    /* Store state in digest */
    memcpy(digest, State, 16);
    /* Zeroize sensitive information. */
    Init();

    return digest;
}

char* MD5::End(char* buf) {
    static const char hex[] = "0123456789abcdef";
    ui8 digest[16];
    if (!buf)
        buf = (char*)malloc(33);
    if (!buf)
        return nullptr;
    Final(digest);
    for (ui8 i = 0; i < MD5_HEX_DIGEST_LENGTH / 2; i++) {
        buf[i * 2] = hex[digest[i] >> 4];
        buf[i * 2 + 1] = hex[digest[i] & 0x0f];
    }
    buf[32] = '\0';
    return buf;
}

char* MD5::End_b64(char* buf) {
    ui8 digest[16];
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
    ui8 digest[16];
    Final(digest);
    ui64 res = 0;
    for (int i = 3; i >= 0; i--) {
        res |= (ui64)(digest[0 + i] ^ digest[8 + i]) << ((3 - i) << 3);
        res |= (ui64)(digest[4 + i] ^ digest[12 + i]) << ((7 - i) << 3);
    }
    return res;
}

TString MD5::Calc(TStringBuf data) {
    return Calc(MakeUnsignedArrayRef(data));
}

TString MD5::Calc(const TArrayRef<const ui8>& data) {
    return Data(data);
}

TString MD5::CalcRaw(TStringBuf data) {
    return CalcRaw(MakeUnsignedArrayRef(data));
}

TString MD5::CalcRaw(const TArrayRef<const ui8>& data) {
    TString result;
    result.ReserveAndResize(16);
    MD5().Update(data).Final(reinterpret_cast<ui8*>(result.begin()));
    return result;
}

ui64 MD5::CalcHalfMix(const char* data, size_t len) {
    return CalcHalfMix(MakeUnsignedArrayRef(data, len));
}

ui64 MD5::CalcHalfMix(TStringBuf data) {
    return CalcHalfMix(MakeUnsignedArrayRef(data));
}

ui64 MD5::CalcHalfMix(const TArrayRef<const ui8>& data) {
    return MD5().Update(data).EndHalfMix();
}

bool MD5::IsMD5(TStringBuf data) {
    return IsMD5(MakeUnsignedArrayRef(data));
}

bool MD5::IsMD5(const TArrayRef<const ui8>& data) {
    if (data.size() != 32) {
        return false;
    }
    for (const ui8 *p = data.data(), *e = data.data() + data.size(); p != e; ++p) {
        if (Char2DigitTable[*p] == '\xff') {
            return false;
        }
    }
    return true;
}
