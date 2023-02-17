#include <unistd.h>

extern const char* ya_get_symbolizer_gen();

const char* ya_get_symbolizer() {
    const char* path = ya_get_symbolizer_gen();
    return access(path, X_OK) ? NULL : path;
}
