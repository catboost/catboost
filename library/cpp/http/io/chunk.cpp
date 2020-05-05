#include "chunk.h"

#include "headers.h"

#include <util/string/cast.h>
#include <util/generic/utility.h>
#include <util/generic/yexception.h>

static inline size_t ParseHex(const TString& s) {
    if (s.empty()) {
        ythrow yexception() << "can not parse chunk length(empty string)";
    }

    size_t ret = 0;

    for (TString::const_iterator c = s.begin(); c != s.end(); ++c) {
        const char ch = *c;

        if (ch >= '0' && ch <= '9') {
            ret *= 16;
            ret += ch - '0';
        } else if (ch >= 'a' && ch <= 'f') {
            ret *= 16;
            ret += 10 + ch - 'a';
        } else if (ch >= 'A' && ch <= 'F') {
            ret *= 16;
            ret += 10 + ch - 'A';
        } else if (ch == ';') {
            break;
        } else if (isspace(ch)) {
            continue;
        } else {
            ythrow yexception() << "can not parse chunk length(" << s.data() << ")";
        }
    }

    return ret;
}

static inline char* ToHex(size_t len, char* buf) {
    do {
        const size_t val = len % 16;

        *--buf = (val < 10) ? (val + '0') : (val - 10 + 'a');
        len /= 16;
    } while (len);

    return buf;
}

class TChunkedInput::TImpl {
public:
    inline TImpl(IInputStream* slave, TMaybe<THttpHeaders>* trailers)
        : Slave_(slave)
        , Trailers_(trailers)
        , Pending_(0)
        , LastChunkReaded_(false)
    {
        if (Trailers_) {
            Trailers_->Clear();
        }
    }

    inline ~TImpl() {
    }

    inline size_t Read(void* buf, size_t len) {
        return Perform(len, [this, buf](size_t toRead) { return Slave_->Read(buf, toRead); });
    }

    inline size_t Skip(size_t len) {
        return Perform(len, [this](size_t toSkip) { return Slave_->Skip(toSkip); });
    }

private:
    template <class Operation>
    inline size_t Perform(size_t len, const Operation& operation) {
        if (!HavePendingData()) {
            return 0;
        }

        const size_t toProcess = Min(Pending_, len);

        if (toProcess) {
            const size_t processed = operation(toProcess);

            if (!processed) {
                ythrow yexception() << "malformed http chunk";
            }

            Pending_ -= processed;

            return processed;
        }

        return 0;
    }

    inline bool HavePendingData() {
        if (LastChunkReaded_) {
            return false;
        }

        if (!Pending_) {
            if (!ProceedToNextChunk()) {
                return false;
            }
        }

        return true;
    }

    inline bool ProceedToNextChunk() {
        TString len(Slave_->ReadLine());

        if (len.empty()) {
            /*
             * skip crlf from previous chunk
             */

            len = Slave_->ReadLine();
        }

        Pending_ = ParseHex(len);

        if (Pending_) {
            return true;
        }

        if (Trailers_) {
            Trailers_->ConstructInPlace(Slave_);
        }
        LastChunkReaded_ = true;

        return false;
    }

private:
    IInputStream* Slave_;
    TMaybe<THttpHeaders>* Trailers_;
    size_t Pending_;
    bool LastChunkReaded_;
};

TChunkedInput::TChunkedInput(IInputStream* slave, TMaybe<THttpHeaders>* trailers)
    : Impl_(new TImpl(slave, trailers))
{
}

TChunkedInput::~TChunkedInput() {
}

size_t TChunkedInput::DoRead(void* buf, size_t len) {
    return Impl_->Read(buf, len);
}

size_t TChunkedInput::DoSkip(size_t len) {
    return Impl_->Skip(len);
}

class TChunkedOutput::TImpl {
    typedef IOutputStream::TPart TPart;

public:
    inline TImpl(IOutputStream* slave)
        : Slave_(slave)
    {
    }

    inline ~TImpl() {
    }

    inline void Write(const void* buf, size_t len) {
        const char* ptr = (const char*)buf;

        while (len) {
            const size_t portion = Min<size_t>(len, 1024 * 16);

            WriteImpl(ptr, portion);

            ptr += portion;
            len -= portion;
        }
    }

    inline void WriteImpl(const void* buf, size_t len) {
        char tmp[32];
        char* e = tmp + sizeof(tmp);
        char* b = ToHex(len, e);

        const TPart parts[] = {
            TPart(b, e - b),
            TPart::CrLf(),
            TPart(buf, len),
            TPart::CrLf(),
        };

        Slave_->Write(parts, sizeof(parts) / sizeof(*parts));
    }

    inline void Flush() {
        Slave_->Flush();
    }

    inline void Finish() {
        Slave_->Write("0\r\n\r\n", 5);

        Flush();
    }

private:
    IOutputStream* Slave_;
};

TChunkedOutput::TChunkedOutput(IOutputStream* slave)
    : Impl_(new TImpl(slave))
{
}

TChunkedOutput::~TChunkedOutput() {
    try {
        Finish();
    } catch (...) {
    }
}

void TChunkedOutput::DoWrite(const void* buf, size_t len) {
    if (Impl_.Get()) {
        Impl_->Write(buf, len);
    } else {
        ythrow yexception() << "can not write to finished stream";
    }
}

void TChunkedOutput::DoFlush() {
    if (Impl_.Get()) {
        Impl_->Flush();
    }
}

void TChunkedOutput::DoFinish() {
    if (Impl_.Get()) {
        Impl_->Finish();
        Impl_.Destroy();
    }
}
