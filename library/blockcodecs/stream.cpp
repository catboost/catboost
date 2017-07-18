#include "stream.h"
#include "codecs.h"

#include <util/digest/murmur.h>
#include <util/generic/cast.h>
#include <util/generic/hash.h>
#include <util/generic/singleton.h>
#include <util/stream/mem.h>
#include <util/ysaveload.h>

using namespace NBlockCodecs;

namespace {
    typedef ui16 TCodecID;
    typedef ui64 TBlockLen;

    struct TIds {
        inline TIds() {
            const TCodecList lst = ListAllCodecs();

            for (size_t i = 0; i < +lst; ++i) {
                const ICodec* c = Codec(lst[i]);

                ByID[CodecID(c)] = c;
            }
        }

        static inline TCodecID CodecID(const ICodec* c) {
            const TStringBuf name = c->Name();

            union {
                ui16 Parts[2];
                ui32 Data;
            } x;

            x.Data = MurmurHash<ui32>(~name, +name);

            return x.Parts[1] ^ x.Parts[0];
        }

        inline const ICodec* Find(TCodecID id) const {
            TByID::const_iterator it = ByID.find(id);

            if (it != ByID.end()) {
                return it->second;
            }

            ythrow yexception() << "can not find codec by id " << id;
        }

        typedef yhash<TCodecID, const ICodec*> TByID;
        TByID ByID;
    };

    static TCodecID CodecID(const ICodec* c) {
        return TIds::CodecID(c);
    }

    static const ICodec* CodecByID(TCodecID id) {
        return Singleton<TIds>()->Find(id);
    }
}

TCodedOutput::TCodedOutput(TOutputStream* out, const ICodec* c, size_t bufLen)
    : C_(c)
    , D_(bufLen)
    , S_(out)
{
}

TCodedOutput::~TCodedOutput() {
    try {
        Finish();
    } catch (...) {
    }
}

void TCodedOutput::DoWrite(const void* buf, size_t len) {
    const char* in = (const char*)buf;

    while (len) {
        const size_t avail = D_.Avail();

        if (len < avail) {
            D_.Append(in, len);

            return;
        }

        D_.Append(in, avail);

        Y_ASSERT(!D_.Avail());

        in += avail;
        len -= avail;

        Y_VERIFY(FlushImpl(), "shit happen");
    }
}

bool TCodedOutput::FlushImpl() {
    const bool ret = !D_.Empty();
    const size_t payload = sizeof(TCodecID) + sizeof(TBlockLen);
    O_.Reserve(C_->MaxCompressedLength(D_) + payload);

    void* out = O_.Data() + payload;
    const size_t olen = C_->Compress(D_, out);

    {
        TMemoryOutput mo(O_.Data(), payload);

        ::Save(&mo, CodecID(C_));
        ::Save(&mo, SafeIntegerCast<TBlockLen>(olen));
    }

    S_->Write(O_.Data(), payload + olen);

    D_.Clear();
    O_.Clear();

    return ret;
}

void TCodedOutput::DoFlush() {
    if (S_ && !D_.Empty()) {
        FlushImpl();
    }
}

void TCodedOutput::DoFinish() {
    if (S_) {
        try {
            if (FlushImpl()) {
                //always write zero-length block as eos marker
                FlushImpl();
            }
        } catch (...) {
            S_ = nullptr;

            throw;
        }

        S_ = nullptr;
    }
}

TDecodedInput::TDecodedInput(TInputStream* in)
    : S_(in)
{
}

TDecodedInput::~TDecodedInput() = default;

size_t TDecodedInput::DoUnboundedNext(const void** ptr) {
    if (!S_) {
        return 0;
    }

    TCodecID codecId;
    TBlockLen blockLen;

    {
        const size_t payload = sizeof(TCodecID) + sizeof(TBlockLen);
        char buf[32];

        S_->LoadOrFail(buf, payload);

        TMemoryInput in(buf, payload);

        ::Load(&in, codecId);
        ::Load(&in, blockLen);
    }

    if (!blockLen) {
        S_ = nullptr;

        return 0;
    }

    TBuffer block;
    block.Resize(blockLen);

    S_->LoadOrFail(block.Data(), blockLen);
    CodecByID(codecId)->Decode(block, D_);

    *ptr = D_.Data();
    return D_.Size();
}
