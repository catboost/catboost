#include "platform.h"
#include "demangle_impl.h"

#ifdef __GNUC__
#include <stdexcept>
#include <cxxabi.h>
#endif

#include "type_name.h"

const char* NPrivate::TCppDemangler::Demangle(const char* name) {
#ifndef __GNUC__
    return name;
#else
    int status;
    TmpBuf_.Reset(__cxxabiv1::__cxa_demangle(name, nullptr, nullptr, &status));

    if (!TmpBuf_) {
        return name;
    }

    return TmpBuf_.Get();
#endif
}

TString CppDemangle(const TString& name) {
    return NPrivate::TCppDemangler().Demangle(name.data());
}

TString TypeName(const std::type_info& typeInfo) {
    return CppDemangle(typeInfo.name());
}

TString TypeName(const std::type_index& typeIndex) {
    return CppDemangle(typeIndex.name());
}
