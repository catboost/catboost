#include "yarchive.h"

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/utility.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>
#include <util/memory/tempbuf.h>
#include <util/stream/input.h>
#include <util/stream/length.h>
#include <util/stream/mem.h>
#include <util/stream/output.h>
#include <util/stream/zlib.h>
#include <util/system/byteorder.h>
#include <util/ysaveload.h>

template <class T>
static inline void ESSave(IOutputStream* out, const T& t_in) {
    T t = HostToLittle(t_in);

    out->Write((const void*)&t, sizeof(t));
}

static inline void ESSave(IOutputStream* out, const TString& s) {
    ESSave(out, (ui32) s.size());
    out->Write(s.data(), s.size());
}

template <class T>
static inline T ESLoad(IInputStream* in) {
    T t = T();

    if (in->Load(&t, sizeof(t)) != sizeof(t)) {
        ythrow TSerializeException() << "malformed archive";
    }

    return LittleToHost(t);
}

template <>
inline TString ESLoad<TString>(IInputStream* in) {
    size_t len = ESLoad<ui32>(in);
    TString ret;
    TTempBuf tmp;

    while (len) {
        const size_t toread = Min(len, tmp.Size());
        const size_t readed = in->Read(tmp.Data(), toread);

        if (!readed) {
            ythrow TSerializeException() << "malformed archive";
        }

        ret.append(tmp.Data(), readed);
        len -= readed;
    }

    return ret;
}

namespace {
    class TArchiveRecordDescriptor: public TSimpleRefCount<TArchiveRecordDescriptor> {
    public:
        inline TArchiveRecordDescriptor(ui64 off, ui64 len, const TString& name)
            : Off_(off)
            , Len_(len)
            , Name_(name)
        {
        }

        inline TArchiveRecordDescriptor(IInputStream* in)
            : Off_(ESLoad<ui64>(in))
            , Len_(ESLoad<ui64>(in))
            , Name_(ESLoad<TString>(in))
        {
        }

        inline ~TArchiveRecordDescriptor() = default;

        inline void SaveTo(IOutputStream* out) const {
            ESSave(out, Off_);
            ESSave(out, Len_);
            ESSave(out, Name_);
        }

        inline const TString& Name() const noexcept {
            return Name_;
        }

        inline ui64 Length() const noexcept {
            return Len_;
        }

        inline ui64 Offset() const noexcept {
            return Off_;
        }

    private:
        ui64 Off_;
        ui64 Len_;
        TString Name_;
    };

    typedef TIntrusivePtr<TArchiveRecordDescriptor> TArchiveRecordDescriptorRef;
}

class TArchiveWriter::TImpl {
    using TDict = THashMap<TString, TArchiveRecordDescriptorRef>;

public:
    inline TImpl(IOutputStream& out, bool compress)
        : Off_(0)
        , Out_(&out)
        , UseCompression(compress)
    {
    }

    inline ~TImpl() = default;

    inline void Flush() {
        Out_->Flush();
    }

    inline void Finish() {
        TCountingOutput out(Out_);

        {
            TZLibCompress compress(&out);

            ESSave(&compress, (ui32)Dict_.size());

            for (const auto& kv : Dict_) {
                kv.second->SaveTo(&compress);
            }

            ESSave(&compress, static_cast<ui8>(UseCompression));

            compress.Finish();
        }

        ESSave(Out_, out.Counter());

        Out_->Flush();
    }

    inline void Add(const TString& key, IInputStream* src) {
        Y_ENSURE(!Dict_.contains(key), "key " << key.data() << " already stored");

        TCountingOutput out(Out_);
        if (UseCompression) {
            TZLibCompress compress(&out);
            TransferData(src, &compress);
            compress.Finish();
        } else {
            size_t skip_size = ArchiveWriterDefaultDataAlignment - Off_ % ArchiveWriterDefaultDataAlignment;
            if (skip_size == ArchiveWriterDefaultDataAlignment) {
                skip_size = 0;
            }
            while(skip_size > 0) {
                Out_->Write(char(0));
                Off_ += 1;
                skip_size -= 1;
            }
            TransferData(src, &out);
            out.Finish();
        }

        TArchiveRecordDescriptorRef descr(new TArchiveRecordDescriptor(Off_, out.Counter(), key));

        Dict_[key] = descr;
        Off_ += out.Counter();
    }

    inline void AddSynonym(const TString& existingKey, const TString& newKey) {
        Y_ENSURE(Dict_.contains(existingKey), "key " << existingKey.data() << " not stored yet");
        Y_ENSURE(!Dict_.contains(newKey), "key " << newKey.data() << " already stored");

        TArchiveRecordDescriptorRef existingDescr = Dict_[existingKey];
        TArchiveRecordDescriptorRef descr(new TArchiveRecordDescriptor(existingDescr->Offset(), existingDescr->Length(), newKey));

        Dict_[newKey] = descr;
    }

private:
    ui64 Off_;
    IOutputStream* Out_;
    TDict Dict_;
    const bool UseCompression;
};

TArchiveWriter::TArchiveWriter(IOutputStream* out, bool compress)
    : Impl_(new TImpl(*out, compress))
{
}

TArchiveWriter::~TArchiveWriter() {
    try {
        Finish();
    } catch (...) {
    }
}

void TArchiveWriter::Flush() {
    if (Impl_.Get()) {
        Impl_->Flush();
    }
}

void TArchiveWriter::Finish() {
    if (Impl_.Get()) {
        Impl_->Finish();
        Impl_.Destroy();
    }
}

void TArchiveWriter::Add(const TString& key, IInputStream* src) {
    Y_ENSURE(Impl_.Get(), "archive already closed");

    Impl_->Add(key, src);
}

void TArchiveWriter::AddSynonym(const TString& existingKey, const TString& newKey) {
    Y_ENSURE(Impl_.Get(), "archive already closed");

    Impl_->AddSynonym(existingKey, newKey);
}

namespace {
    class TArchiveInputStreamBase {
    public:
        inline TArchiveInputStreamBase(const TBlob& b)
            : Blob_(b)
            , Input_(b.Data(), b.Size())
        {
        }

    protected:
        TBlob Blob_;
        TMemoryInput Input_;
    };

    class TArchiveInputStream: public TArchiveInputStreamBase, public TZLibDecompress {
    public:
        inline TArchiveInputStream(const TBlob& b)
            : TArchiveInputStreamBase(b)
            , TZLibDecompress(&Input_)
        {
        }

        ~TArchiveInputStream() override = default;
    };
}

class TArchiveReader::TImpl {
    typedef THashMap<TString, TArchiveRecordDescriptorRef> TDict;

public:
    inline TImpl(const TBlob& blob)
        : Blob_(blob)
        , UseDecompression(true)
    {
        ReadDict();
    }

    inline ~TImpl() = default;

    inline void ReadDict() {
        Y_ENSURE(Blob_.Size() >= sizeof(ui64), "too small blob");

        const char* end = (const char*)Blob_.End();
        const char* ptr = end - sizeof(ui64);
        ui64 dictlen = 0;
        memcpy(&dictlen, ptr, sizeof(ui64));
        dictlen = LittleToHost(dictlen);

        Y_ENSURE(dictlen <= Blob_.Size() - sizeof(ui64), "bad blob");

        const char* beg = ptr - dictlen;
        TMemoryInput mi(beg, dictlen);
        TZLibDecompress d(&mi);
        const ui32 count = ESLoad<ui32>(&d);

        for (size_t i = 0; i < count; ++i) {
            TArchiveRecordDescriptorRef descr(new TArchiveRecordDescriptor(&d));

            Recs_.push_back(descr);
            Dict_[descr->Name()] = descr;
        }
        Sort(Recs_.begin(), Recs_.end(), [](const auto& lhs, const auto& rhs) -> bool {
            return lhs->Offset() < rhs->Offset();
        });

        try {
            UseDecompression = static_cast<bool>(ESLoad<ui8>(&d));
        } catch (const TSerializeException&) {
            // that's ok - just old format
            UseDecompression = true;
        }
    }

    inline size_t Count() const noexcept {
        return Recs_.size();
    }

    inline TString KeyByIndex(size_t n) const {
        if (n < Count()) {
            return Recs_[n]->Name();
        }

        ythrow yexception() << "incorrect index";
    }

    inline bool Has(const TStringBuf key) const {
        return Dict_.contains(key);
    }

    inline TAutoPtr<IInputStream> ObjectByKey(const TStringBuf key) const {
        TBlob subBlob = BlobByKey(key);

        if (UseDecompression) {
            return new TArchiveInputStream(subBlob);
        } else {
            return new TMemoryInput(subBlob.Data(), subBlob.Length());
        }
    }

    inline TBlob ObjectBlobByKey(const TStringBuf key) const {
        TBlob subBlob = BlobByKey(key);

        if (UseDecompression) {
            TArchiveInputStream st(subBlob);
            return TBlob::FromStream(st);
        } else {
            return subBlob;
        }
    }

    inline TBlob BlobByKey(const TStringBuf key) const {
        const auto it = Dict_.find(key);

        Y_ENSURE(it != Dict_.end(), "key " << key.data() << " not found");

        const size_t off = it->second->Offset();
        const size_t len = it->second->Length();

        /*
             * TODO - overflow check
             */

        return Blob_.SubBlob(off, off + len);
    }

    inline bool Compressed() const {
        return UseDecompression;
    }

private:
    TBlob Blob_;
    TVector<TArchiveRecordDescriptorRef> Recs_;
    TDict Dict_;
    bool UseDecompression;
};

TArchiveReader::TArchiveReader(const TBlob& data)
    : Impl_(new TImpl(data))
{
}

TArchiveReader::~TArchiveReader() {}

size_t TArchiveReader::Count() const noexcept {
    return Impl_->Count();
}

TString TArchiveReader::KeyByIndex(size_t n) const {
    return Impl_->KeyByIndex(n);
}

bool TArchiveReader::Has(const TStringBuf key) const {
    return Impl_->Has(key);
}

TAutoPtr<IInputStream> TArchiveReader::ObjectByKey(const TStringBuf key) const {
    return Impl_->ObjectByKey(key);
}

TBlob TArchiveReader::ObjectBlobByKey(const TStringBuf key) const {
    return Impl_->ObjectBlobByKey(key);
}

TBlob TArchiveReader::BlobByKey(const TStringBuf key) const {
    return Impl_->BlobByKey(key);
}

bool TArchiveReader::Compressed() const {
    return Impl_->Compressed();
}
