#include "syms.h"

#undef BEGIN_SYMS
#undef END_SYMS
#undef SYM
#undef ESYM

#include "Python.h"

extern "C" {
#include <library/python/ctypes/syms.h>
}

#include <util/generic/vector.h>
#include <util/generic/string.h>
#include <util/generic/singleton.h>

namespace {
    struct TSym {
        const char* Mod;
        const char* Name;
        void* Sym;
    };

    struct TSymbols: public TVector<TSym> {
        static inline TSymbols* Instance() {
            return Singleton<TSymbols>();
        }
    };
}

namespace NPrivate {
    void RegisterSymbol(const char* mod, const char* name, void* sym) {
        TSymbols::Instance()->push_back(TSym{mod, name, sym});
    }
}

extern "C" {

BEGIN_SYMS() {
    for (auto& x : *TSymbols::Instance()) {
        DictSetStringPtr(d, (TString(x.Mod) + "|" + TString(x.Name)).c_str(), x.Sym);
    }
} END_SYMS()

}
