#include <library/python/symbols/registry/syms.h>

#include <util/generic/guid.h>
#include <util/generic/yexception.h>
#include <util/generic/singleton.h>
#include <util/digest/numeric.h>
#include <util/system/getpid.h>
#include <util/stream/output.h>

#if defined(_unix_)
#include <pthread.h>
#endif

namespace {
    struct TPid {
        ui32 Pid = GetPID();

        inline TPid() noexcept {
            (void)AtFork;

#if defined(_unix_)
            Y_ENSURE(pthread_atfork(0, 0, AtFork) == 0, "it happen");
#endif
        }

        static inline TPid& Instance() noexcept {
            return *Singleton<TPid>();
        }

        static void AtFork() noexcept {
            Instance().Pid = GetPID();
        }
    };

    static int uuid_generate_time(void* out) {
        TGUID g;

        CreateGuid(&g);

        g.dw[3] = IntHash(TPid::Instance().Pid ^ g.dw[3]);

        memcpy(out, g.dw, 16);

        return 0;
    }
}

BEGIN_SYMS("uuid")
SYM(uuid_generate_time)
END_SYMS()
