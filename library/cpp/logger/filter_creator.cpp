#include "filter_creator.h"
#include "filter.h"

TFilteredBackendCreator::TFilteredBackendCreator(THolder<ILogBackendCreator> slave, ELogPriority priority)
    : Slave(std::move(slave))
    , Priority(priority)
{}

THolder<TLogBackend> TFilteredBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TFilteredLogBackend>(Slave->CreateLogBackend(), Priority);
}

bool TFilteredBackendCreator::Init(const IInitContext& ctx) {
    return Slave->Init(ctx);
}

void TFilteredBackendCreator::ToJson(NJson::TJsonValue& value) const {
    value["LogLevel"] = ToString(Priority);
    Slave->ToJson(value);
}
