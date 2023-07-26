#include "symbols.h"

#include <util/generic/singleton.h>
#include <util/generic/utility.h>
#include <util/system/dynlib.h>

#define LOADSYM(name, type) {name = (TId<type>::R*)L->SymOptional(#name);}

const TInfinibandSymbols* IBSym() {
    struct TSymbols: TInfinibandSymbols {
        TSymbols() {
            L.Reset(new TDynamicLibrary("/usr/lib/libibverbs.so"));

            DOVERBS(LOADSYM)
        }

        THolder<TDynamicLibrary> L;
    };

    return SingletonWithPriority<TSymbols, 100>();
}

const TRdmaSymbols* RDSym() {
    struct TSymbols: TRdmaSymbols {
        TSymbols() {
            L.Reset(new TDynamicLibrary("/usr/lib/librdmacm.so"));

            DORDMA(LOADSYM)
        }

        THolder<TDynamicLibrary> L;
    };

    return SingletonWithPriority<TSymbols, 100>();
}

const TMlx5Symbols* M5Sym() {
    struct TSymbols: TMlx5Symbols {
        TSymbols() {
            L.Reset(new TDynamicLibrary("/usr/lib/libmlx5.so"));

            DOMLX5(LOADSYM)
        }

        THolder<TDynamicLibrary> L;
    };

    return SingletonWithPriority<TSymbols, 100>();
}

#undef LOADSYM
