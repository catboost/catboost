#include "symbols.h"

#include <ibdrv/verbs_loader.h>

#include <util/generic/yexception.h>
#include <util/generic/vector.h>
#include <util/generic/singleton.h>
#include <util/generic/utility.h>
#include <util/system/dynlib.h>
#include <util/string/builder.h>
#include <library/cpp/iterator/zip.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#define LOADSYM(name, type) {name = (TId<type>::R*)L->SymOptional(#name);}

#define SET_L_TRYING_PATHS(_paths, _do_some) \
    auto lib = std::make_unique<TDynamicLibrary>(); \
    TVector<TString> paths = _paths; \
    TVector<TString> errorMessages; \
    for (auto path : paths) { \
        try { \
            TString libOpenErrorMessage; \
            if (!lib->TryOpen(path.c_str(), DEFAULT_DLLOPEN_FLAGS, &libOpenErrorMessage)) { \
                errorMessages.emplace_back(std::move(libOpenErrorMessage)); \
                continue; \
            } \
            L.Reset(lib.release()); \
            _do_some \
            return; \
        } catch (std::exception& ex) { \
            errorMessages.emplace_back(ex.what()); \
        } \
    } \
    Y_ABORT_UNLESS(paths.size() == errorMessages.size()); \
    TStringBuilder builder; \
    builder << "Cannot open any shared library. Reasons:\n"; \
    for (const auto& [reason, path] : Zip(errorMessages, paths)) { \
        builder << "Path: " << path << " Reason: " << reason << "\n"; \
    } \
    ythrow yexception() << builder;

#define LIBIBVERBS_PATHS {"/usr/lib/libibverbs.so", "libibverbs.so", "libibverbs.so.1"}

namespace {
    class TInfinibandSymbolsHolder final: public TInfinibandSymbols {
    public:
        TInfinibandSymbolsHolder() {
            SET_L_TRYING_PATHS(LIBIBVERBS_PATHS, DOVERBS(LOADSYM))
        }

        bool HasSymbol(const char* name) const noexcept {
            return L->SymOptional(name) != nullptr;
        }

    private:
        THolder<TDynamicLibrary> L;
    };

    const TInfinibandSymbolsHolder* LoadedIBSymbols() {
        return SingletonWithPriority<TInfinibandSymbolsHolder, 100>();
    }

    void SetLoadError(char* output, size_t outputSize, const char* message) noexcept {
        if (output == nullptr || outputSize == 0 || message == nullptr) {
            return;
        }
        const size_t length = std::min(outputSize - 1, std::strlen(message));
        std::memcpy(output, message, length);
        output[length] = '\0';
    }
}

const TInfinibandSymbols* IBSym() {
    return LoadedIBSymbols();
}

int ibdrv_try_load_ibverbs(
    const char* const* requiredSymbols,
    size_t requiredSymbolCount,
    char* errorMessage,
    size_t errorMessageSize) noexcept {
    if (errorMessage != nullptr && errorMessageSize > 0) {
        errorMessage[0] = '\0';
    }
    if (requiredSymbolCount > 0 && requiredSymbols == nullptr) {
        SetLoadError(errorMessage, errorMessageSize, "required_symbols is null");
        return -EINVAL;
    }

    try {
        const auto* symbols = LoadedIBSymbols();
        for (size_t index = 0; index < requiredSymbolCount; ++index) {
            const char* symbol = requiredSymbols[index];
            if (symbol == nullptr || symbol[0] == '\0') {
                SetLoadError(errorMessage, errorMessageSize, "required symbol name is empty");
                return -EINVAL;
            }
            if (!symbols->HasSymbol(symbol)) {
                const TString message = TStringBuilder()
                    << "Required libibverbs symbol is unavailable: " << symbol;
                SetLoadError(errorMessage, errorMessageSize, message.c_str());
                return -ENOSYS;
            }
        }
        return 0;
    } catch (const yexception& ex) {
        SetLoadError(errorMessage, errorMessageSize, ex.what());
        return -ENOENT;
    } catch (const std::exception& ex) {
        SetLoadError(errorMessage, errorMessageSize, ex.what());
        return -EIO;
    } catch (...) {
        SetLoadError(errorMessage, errorMessageSize, "Unknown libibverbs loading error");
        return -EIO;
    }
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
