#include "system_creator.h"

THolder<TLogBackend> TSysLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TSysLogBackend>(Ident.c_str(), Facility, Flags);
}


TSysLogBackendCreator::TSysLogBackendCreator()
    : TLogBackendCreatorBase("system")
{}

bool TSysLogBackendCreator::Init(const IInitContext& ctx) {
    ctx.GetValue("Ident", Ident);
    ctx.GetValue("Facility", (int&)Facility);
    ctx.GetValue("Flags", Flags);
    return true;
}

ILogBackendCreator::TFactory::TRegistrator<TSysLogBackendCreator> TSysLogBackendCreator::Registrar("system");

void TSysLogBackendCreator::DoToJson(NJson::TJsonValue& value) const {
    value["Ident"] = Ident;
    value["Facility"] = (int&)Facility;
    value["Flags"] = Flags;
}
