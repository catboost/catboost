#pragma once

#include <util/generic/ptr.h>
#include <util/generic/string.h>

class TCppDemangler {
public:
    const char* Demangle(const char* name);

private:
    THolder<char, TFree> TmpBuf_;
};

inline TString CppDemangle(const TString& name) {
    return TCppDemangler().Demangle(name.data());
}
