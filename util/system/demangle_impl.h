#pragma once

#include <util/generic/ptr.h>
#include <util/generic/string.h>

namespace NPrivate {

    /*
     * cxxabi::__cxa_demangle (and thus TCppDemanger) have terrible memory ownership model.
     *
     * Consider using CppDemangle instead. It is slow, but robust.
     */
    class TCppDemangler {
    public:
        const char* Demangle(const char* name);

    private:
        THolder<char, TFree> TmpBuf_;
    };

} // namespace NPrivate
