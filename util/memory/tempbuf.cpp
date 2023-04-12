#include "tempbuf.h"
#include "addstorage.h"

#include <util/system/yassert.h>
#include <util/system/defaults.h>
#include <util/generic/intrlist.h>
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <utility>
#include <util/thread/singleton.h>

#ifndef TMP_BUF_LEN
    #define TMP_BUF_LEN (64 * 1024)
#endif

class TTempBuf::TImpl: public TRefCounted<TImpl, TSimpleCounter, TImpl> {
public:
    inline TImpl(void* data, size_t size) noexcept
        : Data_(data)
        , Size_(size)
        , Offset_(0)
    {
    }

    /*
     * We do not really need 'virtual' here
     * but for compiler happiness...
     */
    virtual ~TImpl() = default;

    inline void* Data() noexcept {
        return Data_;
    }

    const void* Data() const noexcept {
        return Data_;
    }

    inline size_t Size() const noexcept {
        return Size_;
    }

    inline size_t Filled() const noexcept {
        return Offset_;
    }

    inline void Reset() noexcept {
        Offset_ = 0;
    }

    inline size_t Left() const noexcept {
        return Size() - Filled();
    }

    void SetPos(size_t off) {
        Y_ASSERT(off <= Size());
        Offset_ = off;
    }

    inline void Proceed(size_t off) {
        Y_ASSERT(off <= Left());

        Offset_ += off;
    }

    static inline void Destroy(TImpl* This) noexcept {
        This->Dispose();
    }

protected:
    virtual void Dispose() noexcept = 0;

private:
    void* Data_;
    size_t Size_;
    size_t Offset_;
};

namespace {
    class TTempBufManager;

    class TAllocedBuf: public TTempBuf::TImpl, public TAdditionalStorage<TAllocedBuf> {
    public:
        inline TAllocedBuf()
            : TImpl(AdditionalData(), AdditionalDataLength())
        {
        }

        inline ~TAllocedBuf() override = default;

    private:
        void Dispose() noexcept override {
            delete this;
        }
    };

    class TPerThreadedBuf: public TTempBuf::TImpl, public TIntrusiveSListItem<TPerThreadedBuf> {
        friend class TTempBufManager;

    public:
        inline TPerThreadedBuf(TTempBufManager* manager) noexcept
            : TImpl(Data_, sizeof(Data_))
            , Manager_(manager)
        {
        }

        inline ~TPerThreadedBuf() override = default;

    private:
        void Dispose() noexcept override;

    private:
        char Data_[TMP_BUF_LEN];
        TTempBufManager* Manager_;
    };

    class TTempBufManager {
        struct TDelete {
            inline void operator()(TPerThreadedBuf* p) noexcept {
                delete p;
            }
        };

    public:
        inline TTempBufManager() noexcept {
        }

        inline ~TTempBufManager() {
            TDelete deleter;

            Unused_.ForEach(deleter);
        }

        inline TPerThreadedBuf* Acquire() {
            if (!Unused_.Empty()) {
                return Unused_.PopFront();
            }

            return new TPerThreadedBuf(this);
        }

        inline void Return(TPerThreadedBuf* buf) noexcept {
            buf->Reset();
            Unused_.PushFront(buf);
        }

    private:
        TIntrusiveSList<TPerThreadedBuf> Unused_;
    };
}

static inline TTempBufManager* TempBufManager() {
    return FastTlsSingletonWithPriority<TTempBufManager, 2>();
}

static inline TTempBuf::TImpl* AcquireSmallBuffer(size_t size) {
#if defined(_asan_enabled_)
    return new (size) TAllocedBuf();
#else
    Y_UNUSED(size);
    return TempBufManager()->Acquire();
#endif
}

void TPerThreadedBuf::Dispose() noexcept {
    if (Manager_ == TempBufManager()) {
        Manager_->Return(this);
    } else {
        delete this;
    }
}

TTempBuf::TTempBuf()
    : Impl_(AcquireSmallBuffer(TMP_BUF_LEN))
{
}

/*
 * all magick is here:
 * if len <= TMP_BUF_LEN. then we get prealloced per threaded buffer
 * else allocate one in heap
 */
static inline TTempBuf::TImpl* ConstructImpl(size_t len) {
    if (len <= TMP_BUF_LEN) {
        return AcquireSmallBuffer(len);
    }

    return new (len) TAllocedBuf();
}

TTempBuf::TTempBuf(size_t len)
    : Impl_(ConstructImpl(len))
{
}

TTempBuf::TTempBuf(const TTempBuf&) noexcept = default;

TTempBuf::TTempBuf(TTempBuf&& b) noexcept
    : Impl_(std::move(b.Impl_))
{
}

TTempBuf::~TTempBuf() = default;

TTempBuf& TTempBuf::operator=(const TTempBuf& b) noexcept {
    if (this != &b) {
        Impl_ = b.Impl_;
    }

    return *this;
}

TTempBuf& TTempBuf::operator=(TTempBuf&& b) noexcept {
    if (this != &b) {
        Impl_ = std::move(b.Impl_);
    }

    return *this;
}

char* TTempBuf::Data() noexcept {
    return (char*)Impl_->Data();
}

const char* TTempBuf::Data() const noexcept {
    return static_cast<const char*>(Impl_->Data());
}

size_t TTempBuf::Size() const noexcept {
    return Impl_->Size();
}

char* TTempBuf::Current() noexcept {
    return Data() + Filled();
}

const char* TTempBuf::Current() const noexcept {
    return Data() + Filled();
}

size_t TTempBuf::Filled() const noexcept {
    return Impl_->Filled();
}

size_t TTempBuf::Left() const noexcept {
    return Impl_->Left();
}

void TTempBuf::Reset() noexcept {
    Impl_->Reset();
}

void TTempBuf::SetPos(size_t off) {
    Impl_->SetPos(off);
}

char* TTempBuf::Proceed(size_t off) {
    char* ptr = Current();
    Impl_->Proceed(off);
    return ptr;
}

void TTempBuf::Append(const void* data, size_t len) {
    if (len > Left()) {
        ythrow yexception() << "temp buf exhausted(" << Left() << ", " << len << ")";
    }

    memcpy(Current(), data, len);
    Proceed(len);
}

bool TTempBuf::IsNull() const noexcept {
    return !Impl_;
}

#if 0
    #include <util/datetime/cputimer.h>

    #define LEN (1024)

void* allocaFunc() {
    return alloca(LEN);
}

int main() {
    const size_t num = 10000000;
    size_t tmp = 0;

    {
        CTimer t("alloca");

        for (size_t i = 0; i < num; ++i) {
            tmp += (size_t)allocaFunc();
        }
    }

    {
        CTimer t("log buffer");

        for (size_t i = 0; i < num; ++i) {
            TTempBuf buf(LEN);

            tmp += (size_t)buf.Data();
        }
    }

    {
        CTimer t("malloc");

        for (size_t i = 0; i < num; ++i) {
            void* ptr = malloc(LEN);

            tmp += (size_t)ptr;

            free(ptr);
        }
    }

    return 0;
}
#endif
