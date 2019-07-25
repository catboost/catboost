#include <util/system/compat.h>
#include <util/system/compiler.h>
#include <util/system/cpu_id.h>
#include <util/system/platform.h>

#define Y_CPU_ID_ENUMERATE_STARTUP_CHECKS(F) \
    F(SSE42)                                 \
    F(PCLMUL)                                \
    F(AES)                                   \
    F(AVX)                                   \
    F(AVX2)                                  \
    F(FMA)

namespace {
    [[noreturn]] void ReportISAError(const char* isa) {
        err(-1, "This program was compiled for %s which is not supported on your system, exiting...", isa);
    }

#define Y_DEF_NAME(X)                  \
    void Assert##X() noexcept {        \
        if (!NX86::Have##X()) {        \
            ReportISAError(#X);        \
        }                              \
    }

    Y_CPU_ID_ENUMERATE_STARTUP_CHECKS(Y_DEF_NAME)
#undef Y_DEF_NAME

    class TBuildCpuChecker {
    public:
        TBuildCpuChecker() {
            Check();
        }

    private:
        void Check() const noexcept {
#if defined(_fma_)
            AssertFMA();
#elif defined(_avx2_)
            AssertAVX2();
#elif defined(_avx_)
            AssertAVX();
#elif defined(_aes_)
            AssertAES();
#elif defined(_pclmul_)
            AssertPCLMUL();
#elif defined(_sse4_2_)
            AssertSSE42();
#endif

#define Y_DEF_NAME(X) Y_UNUSED(Assert##X);
            Y_CPU_ID_ENUMERATE_STARTUP_CHECKS(Y_DEF_NAME)
#undef Y_DEF_NAME
        }
    };
}

#if defined(_x86_) && !defined(_MSC_VER)
#define INIT_PRIORITY(x) __attribute__((init_priority(x)))
#else
#define INIT_PRIORITY(x)
#endif

const static TBuildCpuChecker CheckCpuWeAreRunningOn INIT_PRIORITY(101) ;
