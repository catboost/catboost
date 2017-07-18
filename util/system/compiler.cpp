#include "compiler.h"
#include <cstdlib>

Y_HIDDEN Y_NO_RETURN void _YandexAbort() {
    std::abort();
}
