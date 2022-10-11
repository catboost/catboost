#include "file.h"
#include "record.h"

#include <util/system/file.h>
#include <util/system/rwlock.h>

/*
 * file log
 */
class TFileLogBackend::TImpl {
public:
    inline TImpl(const TString& path)
        : File_(OpenFile(path))
    {
    }

    inline void WriteData(const TLogRecord& rec) {
        //many writes are thread-safe
        TReadGuard guard(Lock_);

        File_.Write(rec.Data, rec.Len);
    }

    inline void ReopenLog() {
        //but log rotate not thread-safe
        TWriteGuard guard(Lock_);

        File_.LinkTo(OpenFile(File_.GetName()));
    }

private:
    static inline TFile OpenFile(const TString& path) {
        return TFile(path, OpenAlways | WrOnly | ForAppend | Seq | NoReuse);
    }

private:
    TRWMutex Lock_;
    TFile File_;
};

TFileLogBackend::TFileLogBackend(const TString& path)
    : Impl_(new TImpl(path))
{
}

TFileLogBackend::~TFileLogBackend() {
}

void TFileLogBackend::WriteData(const TLogRecord& rec) {
    Impl_->WriteData(rec);
}

void TFileLogBackend::ReopenLog() {
    TAtomicSharedPtr<TImpl> copy = Impl_;
    if (copy) {
        copy->ReopenLog();
    }
}
