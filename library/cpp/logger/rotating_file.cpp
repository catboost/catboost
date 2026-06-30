#include "rotating_file.h"
#include "file.h"
#include "record.h"

#include <util/string/builder.h>
#include <util/system/fstat.h>
#include <util/system/rwlock.h>
#include <util/system/fs.h>

#include <atomic>

/*
 * rotating file log
 * if Size_ > MaxSizeBytes
 *    Path.(N-1) -> Path.N
 *    Path.(N-2) -> Path.(N-1)
 *    ...
 *    Path.1     -> Path.2
 *    Path       -> Path.1
 */
class TRotatingFileLogBackend::TImpl {
public:
    inline TImpl(const TString& path, const ui64 maxSizeBytes, const ui32 rotatedFilesCount)
        : Log_(path)
        , Path_(path)
        , MaxSizeBytes_(maxSizeBytes)
        , Size_(TFileStat(Path_).Size)
        , RotatedFilesCount_(rotatedFilesCount)
    {
        Y_ENSURE(RotatedFilesCount_ != 0);
    }

    void WriteData(const TLogRecord& rec) {
        if (Size_.load() > MaxSizeBytes_) {
            TWriteGuard guard(Lock_);
            if (Y_LIKELY(Size_.load() > MaxSizeBytes_)) {
                TString newLogPath(TStringBuilder{} << Path_ << "." << RotatedFilesCount_);
                for (size_t fileId = RotatedFilesCount_ - 1; fileId; --fileId) {
                    TString oldLogPath(TStringBuilder{} << Path_ << "." << fileId);
                    NFs::Rename(oldLogPath, newLogPath);
                    newLogPath = oldLogPath;
                }
                NFs::Rename(Path_, newLogPath);
                Log_.ReopenLog();
                Size_.store(0);
            }
        }
        TReadGuard guard(Lock_);
        Log_.WriteData(rec);
        Size_ += rec.Len;
    }

    void ReopenLog() {
        TWriteGuard guard(Lock_);

        Log_.ReopenLog();
        Size_.store(TFileStat(Path_).Size);
    }

private:
    TRWMutex Lock_;
    TFileLogBackend Log_;
    const TString Path_;
    const ui64 MaxSizeBytes_;
    std::atomic<ui64> Size_;
    const ui32 RotatedFilesCount_;
};

TRotatingFileLogBackend::TRotatingFileLogBackend(const TString& path, const ui64 maxSizeByte, const ui32 rotatedFilesCount)
    : Impl_(new TImpl(path, maxSizeByte, rotatedFilesCount))
{
}

TRotatingFileLogBackend::~TRotatingFileLogBackend() = default;

void TRotatingFileLogBackend::WriteData(const TLogRecord& rec) {
    Impl_->WriteData(rec);
}

void TRotatingFileLogBackend::ReopenLog() {
    TAtomicSharedPtr<TImpl> copy = Impl_;
    if (copy) {
        copy->ReopenLog();
    }
}
