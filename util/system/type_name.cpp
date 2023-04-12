#include "platform.h"
#include "demangle_impl.h"

#ifdef __GNUC__
    #include <cxxabi.h>
#endif

#include "type_name.h"

namespace {

#if defined(_LIBCPP_VERSION)
    // libc++ is nested under std::__y1
    constexpr std::string_view STD_ABI_PREFIX = "std::__y1::";
#elif defined(_linux_)
    // libstdc++ is nested under std::__cxx11
    // FIXME: is there any way to test if we are building against libstdc++?
    constexpr std::string_view STD_ABI_PREFIX = "std::__cxx11::";
#else
    // There is no need to cutoff ABI prefix on Windows
#endif
    constexpr std::string_view STD_PREFIX = "std::";

} // anonymous namespace

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
    TString demangled = CppDemangle(typeInfo.name()); // NOLINT(arcadia-typeid-name-restriction)
#if defined(_linux_) || defined(_darwin_)
    SubstGlobal(demangled, STD_ABI_PREFIX, STD_PREFIX);
#endif
    return demangled;
}

TString TypeName(const std::type_index& typeIndex) {
    TString demangled = CppDemangle(typeIndex.name());
#if defined(_linux_) || defined(_darwin_)
    SubstGlobal(demangled, STD_ABI_PREFIX, STD_PREFIX);
#endif
    return demangled;
}
