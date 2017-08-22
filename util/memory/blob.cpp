#include "blob.h"
#include "addstorage.h"

#include <util/system/yassert.h>
#include <util/system/filemap.h>
#include <util/stream/output.h>
#include <util/stream/buffer.h>
#include <util/generic/ptr.h>
#include <util/generic/string.h>
#include <util/generic/buffer.h>
#include <util/generic/ylimits.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>

class TNullBlobBase: public TBlob::TBase {
public:
    ~TNullBlobBase() override = default;

    void Ref() noexcept override {
    }

    void UnRef() noexcept override {
    }

    static TNullBlobBase* CommonBase() noexcept;
};

TNullBlobBase* TNullBlobBase::CommonBase() noexcept {
    return SingletonWithPriority<TNullBlobBase, 0>();
}

template <class TCounter>
class TDynamicBlobBase: public TBlob::TBase,
                         public TRefCounted<TDynamicBlobBase<TCounter>, TCounter>,
                         public TAdditionalStorage<TDynamicBlobBase<TCounter>> {
    using TRefBase = TRefCounted<TDynamicBlobBase, TCounter>;

public:
    inline TDynamicBlobBase() = default;

    ~TDynamicBlobBase() override = default;

    void Ref() noexcept override {
        TRefBase::Ref();
    }

    void UnRef() noexcept override {
        TRefBase::UnRef();
    }

    inline void* Data() const noexcept {
        return this->AdditionalData();
    }

    inline size_t Length() const noexcept {
        return this->AdditionalDataLength();
    }
};

template <class TCounter>
class TBufferBlobBase: public TBlob::TBase, public TRefCounted<TBufferBlobBase<TCounter>, TCounter> {
    using TRefBase = TRefCounted<TBufferBlobBase, TCounter>;

public:
    inline TBufferBlobBase(TBuffer& buf) {
        Buf_.Swap(buf);
    }

    ~TBufferBlobBase() override = default;

    void Ref() noexcept override {
        TRefBase::Ref();
    }

    void UnRef() noexcept override {
        TRefBase::UnRef();
    }

    inline const TBuffer& Buffer() const noexcept {
        return Buf_;
    }

private:
    TBuffer Buf_;
};

template <class TCounter>
class TStringBlobBase: public TBlob::TBase, public TRefCounted<TStringBlobBase<TCounter>, TCounter> {
    using TRefBase = TRefCounted<TStringBlobBase, TCounter>;

public:
    inline TStringBlobBase(const TString& s)
        : S_(s)
    {
    }

    ~TStringBlobBase() override = default;

    void Ref() noexcept override {
        TRefBase::Ref();
    }

    void UnRef() noexcept override {
        TRefBase::UnRef();
    }

    inline const TString& String() const noexcept {
        return S_;
    }

private:
    const TString S_;
};

template <class TCounter>
class TMappedBlobBase: public TBlob::TBase, public TRefCounted<TMappedBlobBase<TCounter>, TCounter> {
    using TRefBase = TRefCounted<TMappedBlobBase<TCounter>, TCounter>;

public:
    inline TMappedBlobBase(const TMemoryMap& map, ui64 offset, size_t len)
        : Map_(map)
    {
        Y_ENSURE(Map_.IsOpen(), STRINGBUF("memory map not open"));

        Map_.Map(offset, len);

        if (len && !Map_.Ptr()) { // Ptr is 0 for blob of size 0
            ythrow yexception() << "can not map(" << offset << ", " << len << ")";
        }
    }

    ~TMappedBlobBase() override = default;

    void Ref() noexcept override {
        TRefBase::Ref();
    }

    void UnRef() noexcept override {
        TRefBase::UnRef();
    }

    inline const void* Data() const noexcept {
        return Map_.Ptr();
    }

    inline size_t Length() const noexcept {
        return Map_.MappedSize();
    }

private:
    TFileMap Map_;
};

TBlob::TBlob() noexcept
    : S_(nullptr, 0, TNullBlobBase::CommonBase())
{
}

TBlob TBlob::SubBlob(size_t len) const {
    /*
     * may be slightly optimized
     */

    return SubBlob(0, len);
}

TBlob TBlob::SubBlob(size_t begin, size_t end) const {
    if (begin > Length() || end > Length() || begin > end) {
        ythrow yexception() << "incorrect subblob (" << begin << ", " << end << ", outer length = " << Length() << ")";
    }

    return TBlob(Begin() + begin, end - begin, S_.Base);
}

TBlob TBlob::DeepCopy() const {
    return TBlob::Copy(Data(), Length());
}

template <class TCounter>
static inline TBlob CopyConstruct(const void* data, size_t len) {
    using Base = TDynamicBlobBase<TCounter>;
    THolder<Base> base(new (len) Base);

    Y_ASSERT(base->Length() == len);

    memcpy(base->Data(), data, len);

    TBlob ret(base->Data(), len, base.Get());
    base.Release();

    return ret;
}

TBlob TBlob::CopySingleThreaded(const void* data, size_t length) {
    return CopyConstruct<TSimpleCounter>(data, length);
}

TBlob TBlob::Copy(const void* data, size_t length) {
    return CopyConstruct<TAtomicCounter>(data, length);
}

TBlob TBlob::NoCopy(const void* data, size_t length) {
    return TBlob(data, length, TNullBlobBase::CommonBase());
}

template <class TCounter>
static inline TBlob ConstructFromMap(const TMemoryMap& map, ui64 offset, size_t length) {
    using TBase = TMappedBlobBase<TCounter>;
    THolder<TBase> base(new TBase(map, offset, length));
    TBlob ret(base->Data(), base->Length(), base.Get());
    base.Release();

    return ret;
}

template <class TCounter, bool precharge, class T>
static inline TBlob ConstructAsMap(const T& t) {
    TMemoryMap::EOpenMode mode = precharge ? (TMemoryMap::oRdOnly | TMemoryMap::oPrecharge) : TMemoryMap::oRdOnly;

    TMemoryMap map(t, mode);
    const ui64 toMap = map.Length();

    if (toMap > Max<size_t>()) {
        ythrow yexception() << "can not map whole file(length = " << toMap << ")";
    }

    return ConstructFromMap<TCounter>(map, 0, (size_t)toMap);
}

TBlob TBlob::FromFileSingleThreaded(const TString& path) {
    return ConstructAsMap<TSimpleCounter, false>(path);
}

TBlob TBlob::FromFile(const TString& path) {
    return ConstructAsMap<TAtomicCounter, false>(path);
}

TBlob TBlob::FromFileSingleThreaded(const TFile& file) {
    return ConstructAsMap<TSimpleCounter, false>(file);
}

TBlob TBlob::FromFile(const TFile& file) {
    return ConstructAsMap<TAtomicCounter, false>(file);
}

TBlob TBlob::PrechargedFromFileSingleThreaded(const TString& path) {
    return ConstructAsMap<TSimpleCounter, true>(path);
}

TBlob TBlob::PrechargedFromFile(const TString& path) {
    return ConstructAsMap<TAtomicCounter, true>(path);
}

TBlob TBlob::PrechargedFromFileSingleThreaded(const TFile& file) {
    return ConstructAsMap<TSimpleCounter, true>(file);
}

TBlob TBlob::PrechargedFromFile(const TFile& file) {
    return ConstructAsMap<TAtomicCounter, true>(file);
}

TBlob TBlob::FromMemoryMapSingleThreaded(const TMemoryMap& map, ui64 offset, size_t length) {
    return ConstructFromMap<TSimpleCounter>(map, offset, length);
}

TBlob TBlob::FromMemoryMap(const TMemoryMap& map, ui64 offset, size_t length) {
    return ConstructFromMap<TAtomicCounter>(map, offset, length);
}

template <class TCounter>
static inline TBlob ReadFromFile(const TFile& file, ui64 offset, size_t length) {
    using TBase = TDynamicBlobBase<TCounter>;
    THolder<TBase> base(new (length) TBase);

    Y_ASSERT(base->Length() == length);

    file.Pload(base->Data(), length, offset);

    TBlob ret(base->Data(), length, base.Get());
    base.Release();

    return ret;
}

template <class TCounter>
static inline TBlob ConstructFromFileContent(const TFile& file, ui64 offset, ui64 length) {
    if (length > Max<size_t>()) {
        ythrow yexception() << "can not read whole file(length = " << length << ")";
    }

    return ReadFromFile<TCounter>(file, offset, static_cast<size_t>(length));
}

TBlob TBlob::FromFileContentSingleThreaded(const TString& path) {
    TFile file(path, RdOnly);
    return ConstructFromFileContent<TSimpleCounter>(file, 0, file.GetLength());
}

TBlob TBlob::FromFileContent(const TString& path) {
    TFile file(path, RdOnly);
    return ConstructFromFileContent<TAtomicCounter>(file, 0, file.GetLength());
}

TBlob TBlob::FromFileContentSingleThreaded(const TFile& file) {
    return ConstructFromFileContent<TSimpleCounter>(file, 0, file.GetLength());
}

TBlob TBlob::FromFileContent(const TFile& file) {
    return ConstructFromFileContent<TAtomicCounter>(file, 0, file.GetLength());
}

TBlob TBlob::FromFileContentSingleThreaded(const TFile& file, ui64 offset, size_t length) {
    return ConstructFromFileContent<TSimpleCounter>(file, offset, length);
}

TBlob TBlob::FromFileContent(const TFile& file, ui64 offset, size_t length) {
    return ConstructFromFileContent<TAtomicCounter>(file, offset, length);
}

template <class TCounter>
static inline TBlob ConstructFromBuffer(TBuffer& in) {
    using TBase = TBufferBlobBase<TCounter>;
    THolder<TBase> base(new TBase(in));

    TBlob ret(base->Buffer().Data(), base->Buffer().Size(), base.Get());
    base.Release();

    return ret;
}

template <class TCounter>
static inline TBlob ConstructFromStream(IInputStream& in) {
    TBuffer buf;

    {
        TBufferOutput out(buf);

        TransferData(&in, &out);
    }

    return ConstructFromBuffer<TCounter>(buf);
}

TBlob TBlob::FromStreamSingleThreaded(IInputStream& in) {
    return ConstructFromStream<TSimpleCounter>(in);
}

TBlob TBlob::FromStream(IInputStream& in) {
    return ConstructFromStream<TAtomicCounter>(in);
}

TBlob TBlob::FromBufferSingleThreaded(TBuffer& in) {
    return ConstructFromBuffer<TSimpleCounter>(in);
}

TBlob TBlob::FromBuffer(TBuffer& in) {
    return ConstructFromBuffer<TAtomicCounter>(in);
}

template <class TCounter>
static inline TBlob ConstructFromString(const TString& s) {
    using TBase = TStringBlobBase<TCounter>;
    THolder<TBase> base(new TBase(s));

    TBlob ret(~base->String(), +base->String(), base.Get());
    base.Release();

    return ret;
}

TBlob TBlob::FromStringSingleThreaded(const TString& s) {
    return ConstructFromString<TSimpleCounter>(s);
}

TBlob TBlob::FromString(const TString& s) {
    return ConstructFromString<TAtomicCounter>(s);
}
