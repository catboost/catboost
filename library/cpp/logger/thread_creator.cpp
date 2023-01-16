#include "thread_creator.h"
#include "thread.h"

TOwningThreadedLogBackendCreator::TOwningThreadedLogBackendCreator(THolder<ILogBackendCreator>&& slave)
    : Slave(std::move(slave))
{}

THolder<TLogBackend> TOwningThreadedLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TOwningThreadedLogBackend>(Slave->CreateLogBackend().Release(), QueueLen, QueueOverflowCallback);
}

bool TOwningThreadedLogBackendCreator::Init(const IInitContext& ctx) {
    ctx.GetValue("QueueLen", QueueLen);
    return Slave->Init(ctx);
}


void TOwningThreadedLogBackendCreator::ToJson(NJson::TJsonValue& value) const {
    value["QueueLen"] =  QueueLen;
    value["Threaded"] = true;
    Slave->ToJson(value);
}

void TOwningThreadedLogBackendCreator::SetQueueOverflowCallback(std::function<void()> callback) {
    QueueOverflowCallback = std::move(callback);
}

void TOwningThreadedLogBackendCreator::SetQueueLen(size_t len) {
    QueueLen = len;
}
