#include "rotating_file.h"
#include "file.h"
#include "record.h"

#include <util/system/fstat.h>
#include <util/system/rwlock.h>
#include <util/system/fs.h>
#include <util/system/atomic.h>
#include <util/generic/string.h>

/*
 * rotating file log
 */
class TRotatingFileLogBackend::TImpl {
public:
    inline TImpl(const TString& preRotatePath, const TString& postRotatePath, const ui64 maxSizeBytes)
        : Log_(preRotatePath)
        , PreRotatePath_(preRotatePath)
        , PostRotatePath_(postRotatePath)
        , MaxSizeBytes_(maxSizeBytes)
        , Size_(TFileStat(PreRotatePath_).Size)
    {
    }

    inline void WriteData(const TLogRecord& rec) {
        if (static_cast<ui64>(AtomicGet(Size_)) > MaxSizeBytes_) {
            TWriteGuard guard(Lock_);
            if (static_cast<ui64>(AtomicGet(Size_)) > MaxSizeBytes_) {
                NFs::Rename(PreRotatePath_, PostRotatePath_);
                Log_.ReopenLog();
                AtomicSet(Size_, 0);
            }
        }
        TReadGuard guard(Lock_);
        Log_.WriteData(rec);
        AtomicAdd(Size_, rec.Len);
    }

    inline void ReopenLog() {
        TWriteGuard guard(Lock_);

        Log_.ReopenLog();
        AtomicSet(Size_, TFileStat(PreRotatePath_).Size);
    }

private:
    TRWMutex Lock_;
    TFileLogBackend Log_;
    const TString PreRotatePath_;
    const TString PostRotatePath_;
    const ui64 MaxSizeBytes_;
    TAtomic Size_;
};

TRotatingFileLogBackend::TRotatingFileLogBackend(const TString& preRotatePath, const TString& postRotatePath, const ui64 maxSizeBytes)
    : Impl_(new TImpl(preRotatePath, postRotatePath, maxSizeBytes))
{
}

TRotatingFileLogBackend::~TRotatingFileLogBackend() {
}

void TRotatingFileLogBackend::WriteData(const TLogRecord& rec) {
    Impl_->WriteData(rec);
}

void TRotatingFileLogBackend::ReopenLog() {
    TAtomicSharedPtr<TImpl> copy = Impl_;
    if (copy) {
        copy->ReopenLog();
    }
}
