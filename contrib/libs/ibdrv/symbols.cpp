#include "symbols.h"

#include <util/generic/yexception.h>
#include <util/generic/vector.h>
#include <util/generic/singleton.h>
#include <util/generic/utility.h>
#include <util/system/dynlib.h>
#include <util/string/builder.h>
#include <library/cpp/iterator/zip.h>

#define LOADSYM(name, type) {name = (TId<type>::R*)L->SymOptional(#name);}

const TInfinibandSymbols* IBSym() {
    struct TSymbols: TInfinibandSymbols {
        TSymbols() {
            auto lib = std::make_unique<TDynamicLibrary>();

            TVector<TString> catchedExceptions;
            TVector<TString> paths = {"/usr/lib/libibverbs.so", "libibverbs.so", "libibverbs.so.1"};

            for (auto path : paths) {
                try {
                    lib->Open(path.c_str());
                    L.Reset(lib.release());
                    DOVERBS(LOADSYM)
                    return;
                } catch (std::exception& ex) {
                    catchedExceptions.emplace_back(ex.what());
                }
            }

            Y_ABORT_UNLESS(paths.size() == catchedExceptions.size());

            TStringBuilder builder;

            builder << "Cannot open any shared library. Reasons:\n";
            for (const auto& [reason, path] : Zip(catchedExceptions, paths)) {
                builder << "Path: " << path << " Reason: " << reason << "\n";
            }

            ythrow yexception() << builder;
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
