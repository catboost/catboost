#pragma once

namespace NPrivate {
    struct ICB {
        virtual void Apply(const char* mod, const char* name, void* sym) = 0;
        virtual ~ICB() = default;
    };

    void ForEachSymbol(ICB& cb);
    void RegisterSymbol(const char* mod, const char* name, void* sym);
}

#define BEGIN_SYMS(name)                   \
    namespace {                            \
        static struct TRegister {          \
            const char* ModuleName = name; \
            inline TRegister() {
#define END_SYMS() \
    }              \
    }              \
    REGISTRY;      \
    }

#define SYM_2(n, s) ::NPrivate::RegisterSymbol(this->ModuleName, n, (void*)&s);
#define SYM(s) SYM_2(#s, s);
