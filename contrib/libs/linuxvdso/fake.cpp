#include "interface.h"

size_t NVdso::Enumerate(TSymbol*, size_t) {
    return 0;
}

void* NVdso::Function(const char*, const char*) {
    return nullptr;
}
