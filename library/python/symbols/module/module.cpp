#include <library/python/symbols/registry/syms.h>

#define CAP(x) SYM_2(x, x)

BEGIN_SYMS("_capability")
#if defined(_musl_)
CAP("musl")
#endif
#if defined(_linux_)
CAP("linux")
#endif
#if defined(_darwin_)
CAP("darwin")
#endif
CAP("_sentinel")
END_SYMS()

#undef CAP

using namespace NPrivate;

#undef BEGIN_SYMS
#undef END_SYMS
#undef SYM
#undef ESYM

#include <util/generic/string.h>
#include <util/generic/function.h>

#include "Python.h"

extern "C" {
#include <library/python/ctypes/syms.h>
}

namespace {
    struct TCB: public ICB {
        template <class T>
        inline TCB(T& t)
            : CB(t)
        {
        }

        void Apply(const char* mod, const char* name, void* sym) override {
            CB(mod, name, sym);
        }

        std::function<void(const char*, const char*, void*)> CB;
    };
}

extern "C" {
    BEGIN_SYMS() {
        auto f = [&] (const char* mod, const char* name, void* sym) {
            DictSetStringPtr(d, (TString(mod) + "|" + TString(name)).c_str(), sym);
        };

        TCB cb(f);

        ForEachSymbol(cb);
    } END_SYMS()
}
