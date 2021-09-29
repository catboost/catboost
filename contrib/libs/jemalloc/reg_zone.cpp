#include <util/system/compiler.h>

extern "C" void je_zone_register();

static volatile bool initialized = false;

namespace {
    struct TInit {
        inline TInit() {
            if (!initialized) {
                je_zone_register();
                initialized = true;
            }
        }
    };

    void zone_register() {
        static TInit init;
    }
}

extern "C" {
    void je_assure_zone_register() {
        if (Y_LIKELY(initialized)) {
            return;
        }

        // Even if we have read false "initialized", real init will be syncronized once by
        // Meyers singleton in <anonymous>::register_zone(). We could do a few
        // redundant "initialized" and singleton creation checks, but no more than that.
        zone_register();
    }
}
