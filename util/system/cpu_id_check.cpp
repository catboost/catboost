#include "compiler.h"
#include "cpu_id.h"
#include "platform.h"
#include "yassert.h"
#include "atomic.h"
#include <util/generic/singleton.h>
#include <util/stream/output.h>

#define Y_CPU_ID_ENUMERATE_STARTUP_CHECKS(F) \
    F(SSE42)                                 \
    F(PCLMUL)                                \
    F(AES)                                   \
    F(AVX)                                   \
    F(AVX2)                                  \
    F(FMA)

namespace {
    [[noreturn]] void ReportISAError(const char* isa) {
        Cerr << "This program was compiled for " << isa << " which is not supported on your system, exiting..." << Endl;
        exit(-1);
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
