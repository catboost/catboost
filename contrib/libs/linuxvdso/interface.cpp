#include "interface.h"
#include "original/vdso_support.h"

#ifdef HAVE_VDSO_SUPPORT

size_t NVdso::Enumerate(TSymbol* s, size_t len) {
    if (!len) {
        return 0;
    }

    base::VDSOSupport vdso;

    if (!vdso.IsPresent()) {
        return 0;
    }

    size_t n = 0;

    for (base::VDSOSupport::SymbolIterator it = vdso.begin(); it != vdso.end(); ++it) {
        *s++ = TSymbol(it->name, (void*)it->address);
        ++n;

        if (!--len) {
            break;
        }
    }

    return n;
}

void* NVdso::Function(const char* name, const char* version) {
    base::VDSOSupport::SymbolInfo info;
    // Have to cast away the `const` to make this reinterpret_cast-able to a function pointer.
    return base::VDSOSupport().LookupSymbol(name, version, STT_FUNC, &info) ? (void*) info.address : nullptr;
}

#else

size_t NVdso::Enumerate(TSymbol*, size_t) {
    return 0;
}

void* NVdso::Function(const char*, const char*) {
    return nullptr;
}

#endif
