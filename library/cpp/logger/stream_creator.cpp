#include "stream_creator.h"
#include "stream.h"

THolder<TLogBackend> TCerrLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TStreamLogBackend>(&Cerr);
}


ILogBackendCreator::TFactory::TRegistrator<TCerrLogBackendCreator> TCerrLogBackendCreator::RegistrarCerr("cerr");
ILogBackendCreator::TFactory::TRegistrator<TCerrLogBackendCreator> TCerrLogBackendCreator::RegistrarConsole("console");

THolder<TLogBackend> TCoutLogBackendCreator::DoCreateLogBackend() const {
    return MakeHolder<TStreamLogBackend>(&Cout);
}

ILogBackendCreator::TFactory::TRegistrator<TCoutLogBackendCreator> TCoutLogBackendCreator::Registrar("cout");
