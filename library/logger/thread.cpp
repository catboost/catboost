#include "thread.h"
#include "record.h"

#include <util/thread/queue.h>
#include <util/system/event.h>
#include <util/memory/addstorage.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>

class TThreadedLogBackend::TImpl {
    class TRec: public IObjectInQueue, public TAdditionalStorage<TRec>, public TLogRecord {
    public:
        inline TRec(TImpl* parent, const TLogRecord& rec)
            : TLogRecord(rec.Priority, (const char*)AdditionalData(), rec.Len)
            , Parent_(parent)
        {
            memcpy(AdditionalData(), rec.Data, rec.Len);
        }

        inline ~TRec() override {
        }

    private:
        void Process(void* /*tsr*/) override {
            THolder<TRec> This(this);

            Parent_->Slave_->WriteData(*this);
        }

    private:
        TImpl* Parent_;
    };

    class TReopener: public IObjectInQueue, public Event, public TAtomicRefCount<TReopener> {
    public:
        inline TReopener(TImpl* parent)
            : Parent_(parent)
        {
            Ref();
        }

        inline ~TReopener() override {
        }

    private:
        void Process(void* /*tsr*/) override {
            try {
                Parent_->Slave_->ReopenLog();
            } catch (...) {
            }

            Signal();
            UnRef();
        }

    private:
        TImpl* Parent_;
    };

public:
    inline TImpl(TLogBackend* slave, size_t queuelen)
        : Slave_(slave)
    {
        Queue_.Start(1, queuelen);
    }

    inline ~TImpl() {
        Queue_.Stop();
    }

    inline void WriteData(const TLogRecord& rec) {
        THolder<TRec> obj(new (rec.Len) TRec(this, rec));

        if (!Queue_.Add(obj.Get())) {
            ythrow yexception() << "log queue exhausted";
        }

        obj.Release();
    }

    // Write an emergency message when the memory allocator is corrupted.
    // The TThreadedLogBackend object can't be used after this method is called.
    inline void WriteEmergencyData(const TLogRecord& rec) noexcept {
        Queue_.Stop();
        Slave_->WriteData(rec);
    }

    inline void ReopenLog() {
        TIntrusivePtr<TReopener> reopener(new TReopener(this));

        if (!Queue_.Add(reopener.Get())) {
            reopener->UnRef(); // Ref() was called in constructor
            ythrow yexception() << "log queue exhausted";
        }

        reopener->Wait();
    }

private:
    TLogBackend* Slave_;
    TMtpQueue Queue_;
};

TThreadedLogBackend::TThreadedLogBackend(TLogBackend* slave)
    : Impl_(new TImpl(slave, 0))
{
}

TThreadedLogBackend::TThreadedLogBackend(TLogBackend* slave, size_t queuelen)
    : Impl_(new TImpl(slave, queuelen))
{
}

TThreadedLogBackend::~TThreadedLogBackend() {
}

void TThreadedLogBackend::WriteData(const TLogRecord& rec) {
    Impl_->WriteData(rec);
}

void TThreadedLogBackend::ReopenLog() {
    Impl_->ReopenLog();
}

void TThreadedLogBackend::WriteEmergencyData(const TLogRecord& rec) {
    Impl_->WriteEmergencyData(rec);
}
