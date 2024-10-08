#include "writer.h"
#include "common.h"

#include <library/cpp/blockcodecs/codecs.h>
#include <library/cpp/json/writer/json.h>

#include <util/generic/scope.h>
#include <util/generic/yexception.h>
#include <util/system/byteorder.h>


using namespace NUCompress;

TCodedOutput::TCodedOutput(IOutputStream* out, const NBlockCodecs::ICodec* c, size_t bufLen)
    : C_(c)
    , D_(bufLen)
    , S_(out)
{
    Y_ENSURE_EX(C_, TBadArgumentException() << "Null codec");
    Y_ENSURE_EX(S_, TBadArgumentException() << "Null output stream");
    D_.Resize(bufLen);
    Y_ENSURE_EX(C_->MaxCompressedLength(D_) <= MaxCompressedLen, TBadArgumentException() << "Too big buffer size: " << bufLen);
    D_.Clear();
}

TCodedOutput::~TCodedOutput() {
    try {
        Finish();
    } catch (...) {
    }
}

void TCodedOutput::DoWrite(const void* buf, size_t len) {
    Y_ENSURE(S_, "Stream finished already");
    const char* in = static_cast<const char*>(buf);

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

        FlushImpl();
    }
}

void TCodedOutput::FlushImpl() {
    if (!HdrWritten) {
        NJsonWriter::TBuf jBuf;
        jBuf.BeginObject();
        jBuf.WriteKey("codec");
        jBuf.WriteString(C_->Name());
        jBuf.EndObject();

        TString jStr = jBuf.Str() + '\n';
        const TBlockLen lenToSave = HostToLittle(jStr.length());
        S_->Write(&lenToSave, sizeof(lenToSave));
        S_->Write(jStr.Detach(), jStr.length());
        HdrWritten = true;
    }

    O_.Reserve(C_->MaxCompressedLength(D_));
    const size_t oLen = C_->Compress(D_, O_.Data());
    Y_ASSERT(oLen <= MaxCompressedLen);

    const TBlockLen lenToSave = HostToLittle(oLen);
    S_->Write(&lenToSave, sizeof(lenToSave));
    S_->Write(O_.Data(), oLen);

    D_.Clear();
    O_.Clear();
}

void TCodedOutput::DoFlush() {
    if (S_ && D_) {
        FlushImpl();
    }
}

void TCodedOutput::DoFinish() {
    if (S_) {
        Y_DEFER {
            S_ = nullptr;
        };
        FlushImpl();
        // Write zero-length block as EOF marker.
        FlushImpl();
    }
}
