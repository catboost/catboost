#include "lz.h"

#include <util/system/yassert.h>
#include <util/system/byteorder.h>
#include <util/memory/addstorage.h>
#include <util/generic/buffer.h>
#include <util/generic/utility.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <util/stream/mem.h>

#include <library/cpp/streams/lz/common/compressor.h>

#include <library/cpp/streams/lz/lz4/block.h>
#include <library/cpp/streams/lz/snappy/block.h>

#include <contrib/libs/fastlz/fastlz.h>
#include <contrib/libs/quicklz/quicklz.h>
#include <contrib/libs/minilzo/minilzo.h>

/*
 * MiniLzo
 */
class TMiniLzo {
    class TInit {
    public:
        inline TInit() {
            if (lzo_init() != LZO_E_OK) {
                ythrow yexception() << "can not init lzo engine";
            }
        }
    };

public:
    static const char signature[];

    inline TMiniLzo() {
        Singleton<TInit>();
    }

    inline ~TMiniLzo() {
    }

    static inline size_t Hint(size_t len) noexcept {
        // see SEARCH-2043 and, e.g. examples at
        // http://stackoverflow.com/questions/4235019/how-to-get-lzo-to-work-with-a-file-stream
        return len + (len / 16) + 64 + 3;
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};

const char TMiniLzo::signature[] = "YLZO";

template <size_t N>
class TFixedArray {
public:
    inline TFixedArray() noexcept {
        memset(WorkMem_, 0, sizeof(WorkMem_));
    }

protected:
    char WorkMem_[N];
};

class TMiniLzoCompressor: public TMiniLzo, public TFixedArray<LZO1X_MEM_COMPRESS + 1> {
public:
    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        lzo_uint out = 0;
        lzo1x_1_compress((const lzo_bytep)data, len, (lzo_bytep)ptr, &out, WorkMem_);

        return out;
    }
};

class TMiniLzoDecompressor: public TMiniLzo, public TFixedArray<LZO1X_MEM_DECOMPRESS + 1> {
public:
    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t /*max*/) {
        lzo_uint ret = 0;

        lzo1x_decompress((const lzo_bytep)data, len, (lzo_bytep)ptr, &ret, WorkMem_);

        return ret;
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }
};

DEF_COMPRESSOR(TLzoCompress, TMiniLzoCompressor)
DEF_DECOMPRESSOR(TLzoDecompress, TMiniLzoDecompressor)

/*
 * FastLZ
 */
class TFastLZ {
public:
    static const char signature[];

    static inline size_t Hint(size_t len) noexcept {
        return Max<size_t>((size_t)(len * 1.06), 100);
    }

    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        return fastlz_compress(data, len, ptr);
    }

    inline size_t Decompress(const char* data, size_t len, char* ptr, size_t max) {
        return fastlz_decompress(data, len, ptr, max);
    }

    inline void InitFromStream(IInputStream*) const noexcept {
    }

    static inline bool SaveIncompressibleChunks() noexcept {
        return false;
    }
};

const char TFastLZ::signature[] = "YLZF";

DEF_COMPRESSOR(TLzfCompress, TFastLZ)
DEF_DECOMPRESSOR(TLzfDecompress, TFastLZ)

/*
 * QuickLZ
 */
class TQuickLZBase {
public:
    static const char signature[];

    static inline size_t Hint(size_t len) noexcept {
        return len + 500;
    }

    inline TQuickLZBase()
        : Table_(nullptr)
    {
    }

    inline void Init(unsigned ver, unsigned lev, unsigned mod, unsigned type) {
        Table_ = LzqTable(ver, lev, mod);

        if (!Table_) {
            ythrow yexception() << "unsupported lzq stream(" << ver << ", " << lev << ", " << mod << ")";
        }

        const size_t size = Table_->Setting(3) + Table_->Setting(type);

        Mem_.Reset(::operator new(size));
        memset(Mem_.Get(), 0, size);
    }

    inline bool SaveIncompressibleChunks() const noexcept {
        // we must save incompressible chunks "as is"
        // after compressor run in streaming mode
        return Table_->Setting(3);
    }

protected:
    const TQuickLZMethods* Table_;
    THolder<void> Mem_;
};

const char TQuickLZBase::signature[] = "YLZQ";

class TQuickLZCompress: public TQuickLZBase {
public:
    inline size_t Compress(const char* data, size_t len, char* ptr, size_t /*dstMaxSize*/) {
        return Table_->Compress(data, ptr, len, (char*)Mem_.Get());
    }
};

class TQuickLZDecompress: public TQuickLZBase {
public:
    inline size_t Decompress(const char* data, size_t /*len*/, char* ptr, size_t /*max*/) {
        return Table_->Decompress(data, ptr, (char*)Mem_.Get());
    }

    inline void InitFromStream(IInputStream* in) {
        const ui8 ver = ::GLoad<ui8>(in);
        const ui8 lev = ::GLoad<ui8>(in);
        const ui8 mod = ::GLoad<ui8>(in);

        Init(ver, lev, mod, 2);
    }
};

class TLzqCompress::TImpl: public TCompressorBase<TQuickLZCompress, TImpl> {
public:
    inline TImpl(IOutputStream* out, ui16 blockSize, EVersion ver, unsigned level, EMode mode)
        : TCompressorBase<TQuickLZCompress, TImpl>(out, blockSize)
    {
        memset(AdditionalData(), 0, AdditionalDataLength());

        Init(ver, level, mode, 1);

        Save((ui8)ver);
        Save((ui8)level);
        Save((ui8)mode);
    }
};

TLzqCompress::TLzqCompress(IOutputStream* slave, ui16 blockSize, EVersion ver, unsigned level, EMode mode)
    : Impl_(new (TImpl::Hint(blockSize)) TImpl(slave, blockSize, ver, level, mode))
{
}

DEF_COMPRESSOR_COMMON(TLzqCompress, TQuickLZCompress)
DEF_DECOMPRESSOR(TLzqDecompress, TQuickLZDecompress)

namespace {
    template <class T>
    struct TInputHolder {
        static inline T Set(T t) noexcept {
            return t;
        }
    };

    template <class T>
    struct TInputHolder<TAutoPtr<T>> {
        inline T* Set(TAutoPtr<T> v) noexcept {
            V_ = v;

            return V_.Get();
        }

        TAutoPtr<T> V_;
    };

    // Decompressing input streams without signature verification
    template <class TInput, class TDecompressor>
    class TLzDecompressInput: public TInputHolder<TInput>, public IInputStream {
    public:
        inline TLzDecompressInput(TInput in)
            : Impl_(this->Set(in))
        {
        }

    private:
        size_t DoRead(void* buf, size_t len) override {
            return Impl_.Read(buf, len);
        }

    private:
        TDecompressorBaseImpl<TDecompressor> Impl_;
    };
}

template <class T>
static TAutoPtr<IInputStream> TryOpenLzDecompressorX(const TDecompressSignature& s, T input) {
    if (s.Check<TLZ4>())
        return new TLzDecompressInput<T, TLZ4>(input);

    if (s.Check<TSnappy>())
        return new TLzDecompressInput<T, TSnappy>(input);

    if (s.Check<TMiniLzo>())
        return new TLzDecompressInput<T, TMiniLzoDecompressor>(input);

    if (s.Check<TFastLZ>())
        return new TLzDecompressInput<T, TFastLZ>(input);

    if (s.Check<TQuickLZDecompress>())
        return new TLzDecompressInput<T, TQuickLZDecompress>(input);

    return nullptr;
}

template <class T>
static inline TAutoPtr<IInputStream> TryOpenLzDecompressorImpl(const TStringBuf& signature, T input) {
    if (signature.size() == SIGNATURE_SIZE) {
        TMemoryInput mem(signature.data(), signature.size());
        TDecompressSignature s(&mem);

        return TryOpenLzDecompressorX(s, input);
    }

    return nullptr;
}

template <class T>
static inline TAutoPtr<IInputStream> TryOpenLzDecompressorImpl(T input) {
    TDecompressSignature s(&*input);

    return TryOpenLzDecompressorX(s, input);
}

template <class T>
static inline TAutoPtr<IInputStream> OpenLzDecompressorImpl(T input) {
    TAutoPtr<IInputStream> ret = TryOpenLzDecompressorImpl(input);

    if (!ret) {
        ythrow TDecompressorError() << "Unknown compression format";
    }

    return ret;
}

TAutoPtr<IInputStream> OpenLzDecompressor(IInputStream* input) {
    return OpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenLzDecompressor(IInputStream* input) {
    return TryOpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenLzDecompressor(const TStringBuf& signature, IInputStream* input) {
    return TryOpenLzDecompressorImpl(signature, input);
}

TAutoPtr<IInputStream> OpenOwnedLzDecompressor(TAutoPtr<IInputStream> input) {
    return OpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenOwnedLzDecompressor(TAutoPtr<IInputStream> input) {
    return TryOpenLzDecompressorImpl(input);
}

TAutoPtr<IInputStream> TryOpenOwnedLzDecompressor(const TStringBuf& signature, TAutoPtr<IInputStream> input) {
    return TryOpenLzDecompressorImpl(signature, input);
}
