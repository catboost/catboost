#include "cpu_id.h"
#include "types.h"
#include "platform.h"

#include <util/generic/singleton.h>

#if defined(_win_)
    #include <intrin.h>
    #include <immintrin.h>
#elif defined(_x86_)
    #include <cpuid.h>
#endif

#include <string.h>

#if defined(_x86_) && !defined(_win_)
static ui64 _xgetbv(ui32 xcr) {
    ui32 eax;
    ui32 edx;
    __asm__ volatile(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(xcr));
    return (static_cast<ui64>(edx) << 32) | eax;
}
#endif

bool NX86::CpuId(ui32 op, ui32 subOp, ui32* res) noexcept {
#if defined(_x86_)
    #if defined(_MSC_VER)
    static_assert(sizeof(int) == sizeof(ui32), "ups, something wrong here");
    __cpuidex((int*)res, op, subOp);
    #else
    __cpuid_count(op, subOp, res[0], res[1], res[2], res[3]);
    #endif
    return true;
#else
    (void)op;
    (void)subOp;

    memset(res, 0, 4 * sizeof(ui32));

    return false;
#endif
}

bool NX86::CpuId(ui32 op, ui32* res) noexcept {
#if defined(_x86_)
    #if defined(_MSC_VER)
    static_assert(sizeof(int) == sizeof(ui32), "ups, something wrong here");
    __cpuid((int*)res, op);
    #else
    __cpuid(op, res[0], res[1], res[2], res[3]);
    #endif
    return true;
#else
    (void)op;

    memset(res, 0, 4 * sizeof(ui32));

    return false;
#endif
}

namespace {
    union TX86CpuInfo {
        ui32 Info[4];

        struct {
            ui32 EAX;
            ui32 EBX;
            ui32 ECX;
            ui32 EDX;
        };

        inline TX86CpuInfo(ui32 op) noexcept {
            NX86::CpuId(op, Info);
        }

        inline TX86CpuInfo(ui32 op, ui32 subOp) noexcept {
            NX86::CpuId(op, subOp, Info);
        }
    };

    static_assert(sizeof(TX86CpuInfo) == 16, "please, fix me");
}

// https://en.wikipedia.org/wiki/CPUID
bool NX86::HaveRDTSCP() noexcept {
    return (TX86CpuInfo(0x80000001).EDX >> 27) & 1u;
}

bool NX86::HaveSSE() noexcept {
    return (TX86CpuInfo(0x1).EDX >> 25) & 1u;
}

bool NX86::HaveSSE2() noexcept {
    return (TX86CpuInfo(0x1).EDX >> 26) & 1u;
}

bool NX86::HaveSSE3() noexcept {
    return TX86CpuInfo(0x1).ECX & 1u;
}

bool NX86::HavePCLMUL() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 1) & 1u;
}

bool NX86::HaveSSSE3() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 9) & 1u;
}

bool NX86::HaveSSE41() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 19) & 1u;
}

bool NX86::HaveSSE42() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 20) & 1u;
}

bool NX86::HaveF16C() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 29) & 1u;
}

bool NX86::HavePOPCNT() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 23) & 1u;
}

bool NX86::HaveAES() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 25) & 1u;
}

bool NX86::HaveXSAVE() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 26) & 1u;
}

bool NX86::HaveOSXSAVE() noexcept {
    return (TX86CpuInfo(0x1).ECX >> 27) & 1u;
}

bool NX86::HaveAVX() noexcept {
#if defined(_x86_)
    // http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
    // https://bugs.chromium.org/p/chromium/issues/detail?id=375968
    return HaveOSXSAVE()                           // implies HaveXSAVE()
           && (_xgetbv(0) & 6u) == 6u              // XMM state and YMM state are enabled by OS
           && ((TX86CpuInfo(0x1).ECX >> 28) & 1u); // AVX bit
#else
    return false;
#endif
}

bool NX86::HaveFMA() noexcept {
    return HaveAVX() && ((TX86CpuInfo(0x1).ECX >> 12) & 1u);
}

bool NX86::HaveAVX2() noexcept {
    return HaveAVX() && ((TX86CpuInfo(0x7, 0).EBX >> 5) & 1u);
}

bool NX86::HaveBMI1() noexcept {
    return (TX86CpuInfo(0x7, 0).EBX >> 3) & 1u;
}

bool NX86::HaveBMI2() noexcept {
    return (TX86CpuInfo(0x7, 0).EBX >> 8) & 1u;
}

bool NX86::HaveAVX512F() noexcept {
#if defined(_x86_)
    // https://software.intel.com/en-us/articles/how-to-detect-knl-instruction-support
    return HaveOSXSAVE()                           // implies HaveXSAVE()
           && (_xgetbv(0) & 6u) == 6u              // XMM state and YMM state are enabled by OS
           && ((_xgetbv(0) >> 5) & 7u) == 7u       // ZMM state is enabled by OS
           && TX86CpuInfo(0x0).EAX >= 0x7          // leaf 7 is present
           && ((TX86CpuInfo(0x7).EBX >> 16) & 1u); // AVX512F bit
#else
    return false;
#endif
}

bool NX86::HaveAVX512DQ() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).EBX >> 17) & 1u);
}

bool NX86::HaveRDSEED() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x7, 0).EBX >> 18) & 1u);
}

bool NX86::HaveADX() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x7, 0).EBX >> 19) & 1u);
}

bool NX86::HaveAVX512IFMA() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).EBX >> 21) & 1u);
}

bool NX86::HavePCOMMIT() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x7, 0).EBX >> 22) & 1u);
}

bool NX86::HaveCLFLUSHOPT() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x7, 0).EBX >> 23) & 1u);
}

bool NX86::HaveCLWB() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x7, 0).EBX >> 24) & 1u);
}

bool NX86::HaveAVX512PF() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).EBX >> 26) & 1u);
}

bool NX86::HaveAVX512ER() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).EBX >> 27) & 1u);
}

bool NX86::HaveAVX512CD() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).EBX >> 28) & 1u);
}

bool NX86::HaveSHA() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x7, 0).EBX >> 29) & 1u);
}

bool NX86::HaveAVX512BW() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).EBX >> 30) & 1u);
}

bool NX86::HaveAVX512VL() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).EBX >> 31) & 1u);
}

bool NX86::HavePREFETCHWT1() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x7, 0).ECX >> 0) & 1u);
}

bool NX86::HaveAVX512VBMI() noexcept {
    return HaveAVX512F() && ((TX86CpuInfo(0x7, 0).ECX >> 1) & 1u);
}

bool NX86::HaveRDRAND() noexcept {
    return TX86CpuInfo(0x0).EAX >= 0x7 && ((TX86CpuInfo(0x1).ECX >> 30) & 1u);
}

const char* CpuBrand(ui32* store) noexcept {
    memset(store, 0, 12 * sizeof(*store));

#if defined(_x86_)
    NX86::CpuId(0x80000002, store);
    NX86::CpuId(0x80000003, store + 4);
    NX86::CpuId(0x80000004, store + 8);
#endif

    return (const char*)store;
}

#define Y_DEF_NAME(X)                                               \
    bool NX86::CachedHave##X() noexcept {                           \
        return SingletonWithPriority<TFlagsCache, 0>()->Have##X##_; \
    }
Y_CPU_ID_ENUMERATE_OUTLINED_CACHED_DEFINE(Y_DEF_NAME)
#undef Y_DEF_NAME
