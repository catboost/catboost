#pragma once

namespace NPrivate {
    void RegisterSymbol(const char* mod, const char* name, void* sym);
}

#define BEGIN_SYMS(name)              \
    namespace {                   \
        static struct TRegister { \
        const char* ModuleName = name;  \
            inline TRegister() {

#define END_SYMS() \
    }              \
    }              \
    REGISTRY;      \
    }

#define SYM(s) ::NPrivate::RegisterSymbol(this->ModuleName, #s, (void*)&s);
