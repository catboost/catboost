#include "null_creator.h"
#include "null.h"

THolder<TLogBackend> TNullLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TNullLogBackend>();
}

ILogBackendCreator::TFactory::TRegistrator<TNullLogBackendCreator> TNullLogBackendCreator::RegistrarDevNull("/dev/null");
ILogBackendCreator::TFactory::TRegistrator<TNullLogBackendCreator> TNullLogBackendCreator::RegistrarNull("null");


void TNullLogBackendCreator::DoToJson(NJson::TJsonValue& /*value*/) const {
}

TNullLogBackendCreator::TNullLogBackendCreator()
    : TLogBackendCreatorBase("null")
{}
