#include <util/system/compiler.h>

extern "C" void je_register_zone();

static volatile bool initialized = false;

namespace {
    struct TInit {
        inline TInit() {
            if (!initialized) {
                je_register_zone();
                initialized = true;
            }
        }
    };

    void register_zone() {
        static TInit init;
    }
}

extern "C" {
    void je_assure_register_zone() {
        if (Y_LIKELY(initialized)) {
            return;
        }

        // Even if we have read false "initialized", real init will be syncronized once by
        // Meyers singleton in <anonymous>::register_zone(). We could do a few
        // redundant "initialized" and singleton creation checks, but no more than that.
        register_zone();
    }
}
