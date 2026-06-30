#include "dynlib.h"

#include "guard.h"
#include "mutex.h"
#include <util/generic/singleton.h>
#include <util/generic/yexception.h>

#ifdef _win32_
    #include "winint.h"

    #define DLLOPEN(path, flags) LoadLibrary(path)
    #define DLLCLOSE(hndl) FreeLibrary(hndl)
    #define DLLSYM(hndl, name) GetProcAddress(hndl, name)
#else
    #include <dlfcn.h>

    #ifndef RTLD_GLOBAL
        #define RTLD_GLOBAL (0)
    #endif

using HINSTANCE = void*;

    #define DLLOPEN(path, flags) dlopen(path, flags)
    #define DLLCLOSE(hndl) dlclose(hndl)
    #define DLLSYM(hndl, name) dlsym(hndl, name)
#endif

inline TString DLLERR() {
#ifdef _unix_
    return dlerror();
#endif

#ifdef _win32_
    char* msg = 0;
    DWORD cnt = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                              nullptr, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (char*)&msg, 0, nullptr);
    if (!msg) {
        return "DLLERR() unknown error";
    }
    while (cnt && isspace(msg[cnt - 1])) {
        --cnt;
    }
    TString err(msg, 0, cnt);
    LocalFree(msg);
    return err;
#endif
}

class TDynamicLibrary::TImpl {
private:
    inline TImpl(const char* path, int flags)
        : Module(DLLOPEN(path, flags))
        , Unloadable(true)
    {
        (void)flags;

        if (!Module) {
            ythrow yexception() << DLLERR().data();
        }
    }

    class TCreateMutex: public TMutex {
    };

public:
    static inline TImpl* SafeCreate(const char* path, int flags) {
        auto guard = Guard(*Singleton<TCreateMutex>());

        return new TImpl(path, flags);
    }

    inline ~TImpl() {
        if (Module && Unloadable) {
            DLLCLOSE(Module);
        }
    }

    inline void* SymOptional(const char* name) noexcept {
        return (void*)DLLSYM(Module, name);
    }

    inline void* Sym(const char* name) {
        void* symbol = SymOptional(name);

        if (symbol == nullptr) {
            ythrow yexception() << DLLERR().data();
        }

        return symbol;
    }

    inline void SetUnloadable(bool unloadable) {
        Unloadable = unloadable;
    }

private:
    HINSTANCE Module;
    bool Unloadable;
};

TDynamicLibrary::TDynamicLibrary() noexcept {
}

TDynamicLibrary::TDynamicLibrary(const TString& path, int flags) {
    Open(path.data(), flags);
}

TDynamicLibrary::~TDynamicLibrary() = default;

void TDynamicLibrary::Open(const char* path, int flags) {
    Impl_.Reset(TImpl::SafeCreate(path, flags));
}

void TDynamicLibrary::Close() noexcept {
    Impl_.Destroy();
}

void* TDynamicLibrary::SymOptional(const char* name) noexcept {
    if (!IsLoaded()) {
        return nullptr;
    }

    return Impl_->SymOptional(name);
}

void* TDynamicLibrary::Sym(const char* name) {
    if (!IsLoaded()) {
        ythrow yexception() << "library not loaded";
    }

    return Impl_->Sym(name);
}

bool TDynamicLibrary::IsLoaded() const noexcept {
    return (bool)Impl_.Get();
}

void TDynamicLibrary::SetUnloadable(bool unloadable) {
    Impl_->SetUnloadable(unloadable);
}
