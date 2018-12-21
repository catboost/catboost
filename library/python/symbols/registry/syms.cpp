#include "syms.h"

#include <util/generic/vector.h>
#include <util/generic/string.h>
#include <util/generic/singleton.h>

using namespace NPrivate;

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

void NPrivate::RegisterSymbol(const char* mod, const char* name, void* sym) {
    TSymbols::Instance()->push_back(TSym{mod, name, sym});
}

void NPrivate::ForEachSymbol(ICB& cb) {
    for (auto& x : *TSymbols::Instance()) {
        cb.Apply(x.Mod, x.Name, x.Sym);
    }
}
