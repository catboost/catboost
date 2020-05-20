#include "thread.h"
#include "record.h"

#include <util/thread/pool.h>
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

    class TReopener: public IObjectInQueue, public TSystemEvent, public TAtomicRefCount<TReopener> {
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
    inline TImpl(TLogBackend* slave, size_t queuelen, std::function<void()> queueOverflowCallback = {})
        : Slave_(slave)
        , QueueOverflowCallback_(std::move(queueOverflowCallback))
    {
        Queue_.Start(1, queuelen);
    }

    inline ~TImpl() {
        Queue_.Stop();
    }

    inline void WriteData(const TLogRecord& rec) {
        THolder<TRec> obj(new (rec.Len) TRec(this, rec));

        if (Queue_.Add(obj.Get())) {
            Y_UNUSED(obj.Release());
            return;
        }

        if (QueueOverflowCallback_) {
            QueueOverflowCallback_();
        } else {
            ythrow yexception() << "log queue exhausted";
        }
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

    inline void ReopenLogNoFlush() {
        Slave_->ReopenLogNoFlush();
    }

    inline size_t QueueSize() const {
        return Queue_.Size();
    }

private:
    TLogBackend* Slave_;
    TThreadPool Queue_{"ThreadedLogBack"};
    const std::function<void()> QueueOverflowCallback_;
};

TThreadedLogBackend::TThreadedLogBackend(TLogBackend* slave)
    : Impl_(new TImpl(slave, 0))
{
}

TThreadedLogBackend::TThreadedLogBackend(TLogBackend* slave, size_t queuelen, std::function<void()> queueOverflowCallback)
    : Impl_(new TImpl(slave, queuelen, std::move(queueOverflowCallback)))
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

void TThreadedLogBackend::ReopenLogNoFlush() {
    Impl_->ReopenLogNoFlush();
}

void TThreadedLogBackend::WriteEmergencyData(const TLogRecord& rec) {
    Impl_->WriteEmergencyData(rec);
}

size_t TThreadedLogBackend::QueueSize() const {
    return Impl_->QueueSize();
}

TOwningThreadedLogBackend::TOwningThreadedLogBackend(TLogBackend* slave)
    : THolder<TLogBackend>(slave)
    , TThreadedLogBackend(Get())
{
}

TOwningThreadedLogBackend::TOwningThreadedLogBackend(TLogBackend* slave, size_t queuelen, std::function<void()> queueOverflowCallback)
    : THolder<TLogBackend>(slave)
    , TThreadedLogBackend(Get(), queuelen, std::move(queueOverflowCallback))
{
}

TOwningThreadedLogBackend::~TOwningThreadedLogBackend() {
}
