#include <stddef.h>
#include <stdio.h>

#ifndef Y_UNUSED
#define Y_UNUSED(var) (void)(var)
#endif

static class Informer {
public:
    Informer() {
        fprintf(stderr, "WARNING: Binary built without instrumentation module"
            " - see https://docs.yandex-team.ru/ya-make/manual/tests/fuzzing for proper build command\n");
        fflush(stderr);
    }
} informer;

extern "C" {

void __sanitizer_set_death_callback(void (*callback)(void)) {
    Y_UNUSED(callback);
}

void __sanitizer_reset_coverage(void) {
}

void __sanitizer_update_counter_bitset_and_clear_counters(size_t) {
}

size_t __sanitizer_get_number_of_counters(void) {
    return 0;
}

size_t __sanitizer_get_total_unique_coverage(void) {
    return 0;
}

} // extern "C"
