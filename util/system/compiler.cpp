#include "compiler.h"
#include <cstdlib>

[[noreturn]] Y_HIDDEN void _YandexAbort() {
    std::abort();
}
