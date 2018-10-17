#include <library/python/symbols/registry/syms.h>

#include <util/generic/guid.h>
#include <util/digest/numeric.h>

namespace {
    static int uuid_generate_time(void* out) {
        TGUID g;

        CreateGuid(&g);
        g.dw[3] = IntHash(g.dw[3] ^ g.dw[2]);
        memcpy(out, g.dw, 16);

        return 0;
    }
}

BEGIN_SYMS("uuid")
SYM(uuid_generate_time)
END_SYMS()
