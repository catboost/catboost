#include <absl/debugging/stacktrace.h>

namespace {
    static struct TInitUnwinder {
        TInitUnwinder() {
            absl::SetStackUnwinder(DummyUnwinder);
        }

        static int DummyUnwinder(void**, int*, int, int, const void*, int*) {
            return 0;
        }
    } init;
}
