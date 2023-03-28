#include "sync_page_cache_file.h"
#include "record.h"

#include <util/generic/buffer.h>
#include <util/generic/yexception.h>
#include <util/system/file.h>
#include <util/system/info.h>
#include <util/system/mutex.h>
#include <util/system/align.h>

class TSyncPageCacheFileLogBackend::TImpl: public TNonCopyable {
public:
    TImpl(const TString& path, size_t maxBufferSize, size_t maxPendingCacheSize)
        : File_{OpenFile(path)}
        , MaxBufferSize_{maxBufferSize}
        , MaxPendingCacheSize_{maxPendingCacheSize}
        , Buffer_{maxBufferSize}
    {
        ResetPtrs();
    }

    ~TImpl() noexcept {
        try {
            Write();
            FlushSync(GuaranteedWrittenPtr_, WrittenPtr_);
        } catch (...) {
        }
    }

    void WriteData(const TLogRecord& rec) {
        TGuard guard{Lock_};

        Buffer_.Append(rec.Data, rec.Len);
        if (Buffer_.size() >= MaxBufferSize_) {
            const i64 prevAlignedEndPtr = PageAlignedWrittenPtr_;
            Write();

            if (prevAlignedEndPtr < PageAlignedWrittenPtr_) {
                FlushAsync(prevAlignedEndPtr, PageAlignedWrittenPtr_);
            }

            const i64 minPendingCacheOffset = PageAlignedWrittenPtr_ - MaxPendingCacheSize_;
            if (minPendingCacheOffset > GuaranteedWrittenPtr_) {
                FlushSync(GuaranteedWrittenPtr_, minPendingCacheOffset);
            }
        }
    }

    void ReopenLog() {
        TGuard guard{Lock_};

        Write();
        FlushSync(GuaranteedWrittenPtr_, WrittenPtr_);

        File_.LinkTo(OpenFile(File_.GetName()));

        ResetPtrs();
    }

private:
    void ResetPtrs() {
        WrittenPtr_ = File_.GetLength();
        PageAlignedWrittenPtr_ = AlignDown(WrittenPtr_, GetPageSize());
        GuaranteedWrittenPtr_ = WrittenPtr_;
    }

    static TFile OpenFile(const TString& path) {
        return TFile{path, OpenAlways | WrOnly | ForAppend | Seq | NoReuse};
    }

    static i64 GetPageSize() {
        static const i64 pageSize = NSystemInfo::GetPageSize();
        Y_ASSUME(IsPowerOf2(pageSize));
        return pageSize;
    }

    void Write() {
        try {
            File_.Write(Buffer_.Data(), Buffer_.Size());
            WrittenPtr_ += Buffer_.Size();
            PageAlignedWrittenPtr_ = AlignDown(WrittenPtr_, GetPageSize());
            Buffer_.Clear();
        } catch (TFileError&) {
            Buffer_.Clear();
            throw;
        }
    }

    void FlushAsync(const i64 from, const i64 to) {
        File_.FlushCache(from, to - from, /* wait = */ false);
    }

    void FlushSync(const i64 from, const i64 to) {
        const i64 begin = AlignDown(from, GetPageSize());
        const i64 end = AlignUp(to, GetPageSize());
        const i64 length = end - begin;

        File_.FlushCache(begin, length, /* wait = */ true);
        File_.EvictCache(begin, length);

        GuaranteedWrittenPtr_ = to;
    }

private:
    TMutex Lock_;
    TFile File_;

    const size_t MaxBufferSize_ = 0;
    const size_t MaxPendingCacheSize_ = 0;

    TBuffer Buffer_;
    i64 WrittenPtr_ = 0;
    i64 PageAlignedWrittenPtr_ = 0;
    i64 GuaranteedWrittenPtr_ = 0;
};

TSyncPageCacheFileLogBackend::TSyncPageCacheFileLogBackend(const TString& path, size_t maxBufferSize, size_t maxPengingCacheSize)
    : Impl_(MakeHolder<TImpl>(path, maxBufferSize, maxPengingCacheSize))
{}

TSyncPageCacheFileLogBackend::~TSyncPageCacheFileLogBackend() {
}

void TSyncPageCacheFileLogBackend::WriteData(const TLogRecord& rec) {
    Impl_->WriteData(rec);
}

void TSyncPageCacheFileLogBackend::ReopenLog() {
    Impl_->ReopenLog();
}
