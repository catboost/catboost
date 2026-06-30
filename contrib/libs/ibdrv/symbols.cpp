#include "symbols.h"

#include <util/generic/yexception.h>
#include <util/generic/vector.h>
#include <util/generic/singleton.h>
#include <util/generic/utility.h>
#include <util/system/dynlib.h>
#include <util/string/builder.h>
#include <library/cpp/iterator/zip.h>

#define LOADSYM(name, type) {name = (TId<type>::R*)L->SymOptional(#name);}

#define SET_L_TRYING_PATHS(_paths, _do_some) \
    auto lib = std::make_unique<TDynamicLibrary>(); \
    TVector<TString> paths = _paths; \
    TVector<TString> catchedExceptions; \
    for (auto path : paths) { \
        try { \
            lib->Open(path.c_str()); \
            L.Reset(lib.release()); \
            _do_some \
            return; \
        } catch (std::exception& ex) { \
            catchedExceptions.emplace_back(ex.what()); \
        } \
    } \
    Y_ABORT_UNLESS(paths.size() == catchedExceptions.size()); \
    TStringBuilder builder; \
    builder << "Cannot open any shared library. Reasons:\n"; \
    for (const auto& [reason, path] : Zip(catchedExceptions, paths)) { \
        builder << "Path: " << path << " Reason: " << reason << "\n"; \
    } \
    ythrow yexception() << builder;

#define LIBIBVERBS_PATHS {"/usr/lib/libibverbs.so", "libibverbs.so", "libibverbs.so.1"}

const TInfinibandSymbols* IBSym() {
    struct TSymbols: TInfinibandSymbols {
        TSymbols() {
            SET_L_TRYING_PATHS(LIBIBVERBS_PATHS, DOVERBS(LOADSYM))
        }

        THolder<TDynamicLibrary> L;
    };

    return SingletonWithPriority<TSymbols, 100>();
}

#define LIBRDMACM_PATHS {"/usr/lib/librdmacm.so", "librdmacm.so"}

const TRdmaSymbols* RDSym() {
    struct TSymbols: TRdmaSymbols {
        TSymbols() {
            SET_L_TRYING_PATHS(LIBRDMACM_PATHS, DORDMA(LOADSYM))
        }

        THolder<TDynamicLibrary> L;
    };

    return SingletonWithPriority<TSymbols, 100>();
}

#define LIBMLX5_PATHS {"/usr/lib/libmlx5.so", "libmlx5.so"}

const TMlx5Symbols* M5Sym() {
    struct TSymbols: TMlx5Symbols {
        TSymbols() {
            SET_L_TRYING_PATHS(LIBMLX5_PATHS, DOMLX5(LOADSYM))
        }

        THolder<TDynamicLibrary> L;
    };

    return SingletonWithPriority<TSymbols, 100>();
}

#undef LOADSYM
