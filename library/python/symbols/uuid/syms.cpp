#include <library/python/symbols/registry/syms.h>

#include <util/generic/guid.h>
#include <util/digest/numeric.h>

namespace {
    static int uuid_generate_time(void* out) {
        CreateGuid((TGUID*)out);

        return 0;
    }
}

BEGIN_SYMS("uuid")
SYM(uuid_generate_time)
END_SYMS()
