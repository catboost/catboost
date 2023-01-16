#include "backend_creator.h"
#include "stream.h"
#include "uninitialized_creator.h"
#include <util/system/yassert.h>
#include <util/stream/debug.h>
#include <util/stream/output.h>


THolder<TLogBackend> ILogBackendCreator::CreateLogBackend() const {
    try {
        return DoCreateLogBackend();
    } catch(...) {
        Cdbg << "Warning: " << CurrentExceptionMessage() << ". Use stderr instead." << Endl;
    }
    return MakeHolder<TStreamLogBackend>(&Cerr);
}

bool ILogBackendCreator::Init(const IInitContext& /*ctx*/) {
    return true;
}

THolder<ILogBackendCreator> ILogBackendCreator::Create(const IInitContext& ctx) {
    auto res = MakeHolder<TLogBackendCreatorUninitialized>();
    if(!res->Init(ctx)) {
        Cdbg << "Cannot init log backend creator";
        return nullptr;
    }
    return res;
}
