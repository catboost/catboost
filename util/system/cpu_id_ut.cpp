#include "cpu_id.h"

#include "platform.h"

#include <library/cpp/testing/unittest/registar.h>

// There are no tests yet for instructions that use 512-bit wide registers because they are not
// supported by some compilers yet.
// Relevant review in LLVM https://reviews.llvm.org/D16757, we should wait untill it will be in our
// version of Clang.
//
// There are also no tests for PREFETCHWT1, PCOMMIT, CLFLUSHOPT and CLWB as they are not supported
// by our compilers yet (and there are no available processors yet :).

static void ExecuteSSEInstruction();
static void ExecuteSSE2Instruction();
static void ExecuteSSE3Instruction();
static void ExecuteSSSE3Instruction();
static void ExecuteSSE41Instruction();
static void ExecuteSSE42Instruction();
static void ExecuteF16CInstruction();
static void ExecuteAVXInstruction();
static void ExecuteAVX2Instruction();
static void ExecutePOPCNTInstruction();
static void ExecuteBMI1Instruction();
static void ExecuteBMI2Instruction();
static void ExecutePCLMULInstruction();
static void ExecuteAESInstruction();
static void ExecuteAVXInstruction();
static void ExecuteAVX2Instruction();
static void ExecuteAVX512FInstruction();
static void ExecuteAVX512DQInstruction();
static void ExecuteAVX512IFMAInstruction();
static void ExecuteAVX512PFInstruction();
static void ExecuteAVX512ERInstruction();
static void ExecuteAVX512CDInstruction();
static void ExecuteAVX512BWInstruction();
static void ExecuteAVX512VLInstruction();
static void ExecuteAVX512VBMIInstruction();
static void ExecutePREFETCHWT1Instruction();
static void ExecuteSHAInstruction();
static void ExecuteADXInstruction();
static void ExecuteRDRANDInstruction();
static void ExecuteRDSEEDInstruction();
static void ExecutePCOMMITInstruction();
static void ExecuteCLFLUSHOPTInstruction();
static void ExecuteCLWBInstruction();

static void ExecuteFMAInstruction() {
}

static void ExecuteRDTSCPInstruction() {
}

static void ExecuteXSAVEInstruction() {
}

static void ExecuteOSXSAVEInstruction() {
}

Y_UNIT_TEST_SUITE(TestCpuId) {
#define DECLARE_TEST_HAVE_INSTRUCTION(name) \
    Y_UNIT_TEST(Test##Have##name) {         \
        if (NX86::Have##name()) {           \
            Execute##name##Instruction();   \
        }                                   \
    }

    Y_CPU_ID_ENUMERATE(DECLARE_TEST_HAVE_INSTRUCTION)
#undef DECLARE_TEST_HAVE_INSTRUCTION

    Y_UNIT_TEST(TestSSE2) {
#if defined(_x86_64_)
        UNIT_ASSERT(NX86::HaveSSE2());
#endif
    }

    Y_UNIT_TEST(TestCpuBrand) {
        ui32 store[12];

        // Cout << CpuBrand(store) << Endl;;

        UNIT_ASSERT(strlen(CpuBrand(store)) > 0);
    }

    Y_UNIT_TEST(TestCachedAndNoncached) {
#define Y_DEF_NAME(X) UNIT_ASSERT_VALUES_EQUAL(NX86::Have##X(), NX86::CachedHave##X());
        Y_CPU_ID_ENUMERATE(Y_DEF_NAME)
#undef Y_DEF_NAME
    }
} // Y_UNIT_TEST_SUITE(TestCpuId)

#if defined(_x86_64_)
    #if defined(__GNUC__)
void ExecuteSSEInstruction() {
    __asm__ __volatile__("xorps %%xmm0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteSSE2Instruction() {
    __asm__ __volatile__("psrldq $0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteSSE3Instruction() {
    __asm__ __volatile__("addsubpd %%xmm0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteSSSE3Instruction() {
    __asm__ __volatile__("psignb %%xmm0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteSSE41Instruction() {
    __asm__ __volatile__("pmuldq %%xmm0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteSSE42Instruction() {
    __asm__ __volatile__("crc32 %%eax, %%eax\n"
                         :
                         :
                         : "eax");
}

void ExecuteF16CInstruction() {
    __asm__ __volatile__("vcvtph2ps %%xmm0, %%ymm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteAVXInstruction() {
    __asm__ __volatile__("vzeroupper\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteAVX2Instruction() {
    __asm__ __volatile__("vpunpcklbw %%ymm0, %%ymm0, %%ymm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecutePOPCNTInstruction() {
    __asm__ __volatile__("popcnt %%eax, %%eax\n"
                         :
                         :
                         : "eax");
}

void ExecuteBMI1Instruction() {
    __asm__ __volatile__("tzcnt %%eax, %%eax\n"
                         :
                         :
                         : "eax");
}

void ExecuteBMI2Instruction() {
    __asm__ __volatile__("pdep %%rax, %%rdi, %%rax\n"
                         :
                         :
                         : "rax");
}

void ExecutePCLMULInstruction() {
    __asm__ __volatile__("pclmullqlqdq %%xmm0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteAESInstruction() {
    __asm__ __volatile__("aesimc %%xmm0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteAVX512FInstruction() {
}

void ExecuteAVX512DQInstruction() {
}

void ExecuteAVX512IFMAInstruction() {
}

void ExecuteAVX512PFInstruction() {
}

void ExecuteAVX512ERInstruction() {
}

void ExecuteAVX512CDInstruction() {
}

void ExecuteAVX512BWInstruction() {
}

void ExecuteAVX512VLInstruction() {
}

void ExecuteAVX512VBMIInstruction() {
}

void ExecutePREFETCHWT1Instruction() {
}

void ExecuteSHAInstruction() {
    __asm__ __volatile__("sha1msg1 %%xmm0, %%xmm0\n"
                         :
                         :
                         : "xmm0");
}

void ExecuteADXInstruction() {
    __asm__ __volatile__("adcx %%eax, %%eax\n"
                         :
                         :
                         : "eax");
}

void ExecuteRDRANDInstruction() {
    __asm__ __volatile__("rdrand %%eax"
                         :
                         :
                         : "eax");
}

void ExecuteRDSEEDInstruction() {
    __asm__ __volatile__("rdseed %%eax"
                         :
                         :
                         : "eax");
}

void ExecutePCOMMITInstruction() {
}

void ExecuteCLFLUSHOPTInstruction() {
}

void ExecuteCLWBInstruction() {
}

    #elif defined(_MSC_VER)
void ExecuteSSEInstruction() {
}

void ExecuteSSE2Instruction() {
}

void ExecuteSSE3Instruction() {
}

void ExecuteSSSE3Instruction() {
}

void ExecuteSSE41Instruction() {
}

void ExecuteSSE42Instruction() {
}

void ExecuteF16CInstruction() {
}

void ExecuteAVXInstruction() {
}

void ExecuteAVX2Instruction() {
}

void ExecutePOPCNTInstruction() {
}

void ExecuteBMI1Instruction() {
}

void ExecuteBMI2Instruction() {
}

void ExecutePCLMULInstruction() {
}

void ExecuteAESInstruction() {
}

void ExecuteAVX512FInstruction() {
}

void ExecuteAVX512DQInstruction() {
}

void ExecuteAVX512IFMAInstruction() {
}

void ExecuteAVX512PFInstruction() {
}

void ExecuteAVX512ERInstruction() {
}

void ExecuteAVX512CDInstruction() {
}

void ExecuteAVX512BWInstruction() {
}

void ExecuteAVX512VLInstruction() {
}

void ExecuteAVX512VBMIInstruction() {
}

void ExecutePREFETCHWT1Instruction() {
}

void ExecuteSHAInstruction() {
}

void ExecuteADXInstruction() {
}

void ExecuteRDRANDInstruction() {
}

void ExecuteRDSEEDInstruction() {
}

void ExecutePCOMMITInstruction() {
}

void ExecuteCLFLUSHOPTInstruction() {
}

void ExecuteCLWBInstruction() {
}

    #else
        #error "unknown compiler"
    #endif
#else
void ExecuteSSEInstruction() {
}

void ExecuteSSE2Instruction() {
}

void ExecuteSSE3Instruction() {
}

void ExecuteSSSE3Instruction() {
}

void ExecuteSSE41Instruction() {
}

void ExecuteSSE42Instruction() {
}

void ExecuteF16CInstruction() {
}

void ExecuteAVXInstruction() {
}

void ExecuteAVX2Instruction() {
}

void ExecutePOPCNTInstruction() {
}

void ExecuteBMI1Instruction() {
}

void ExecuteBMI2Instruction() {
}

void ExecutePCLMULInstruction() {
}

void ExecuteAESInstruction() {
}

void ExecuteAVX512FInstruction() {
}

void ExecuteAVX512DQInstruction() {
}

void ExecuteAVX512IFMAInstruction() {
}

void ExecuteAVX512PFInstruction() {
}

void ExecuteAVX512ERInstruction() {
}

void ExecuteAVX512CDInstruction() {
}

void ExecuteAVX512BWInstruction() {
}

void ExecuteAVX512VLInstruction() {
}

void ExecuteAVX512VBMIInstruction() {
}

void ExecutePREFETCHWT1Instruction() {
}

void ExecuteSHAInstruction() {
}

void ExecuteADXInstruction() {
}

void ExecuteRDRANDInstruction() {
}

void ExecuteRDSEEDInstruction() {
}

void ExecutePCOMMITInstruction() {
}

void ExecuteCLFLUSHOPTInstruction() {
}

void ExecuteCLWBInstruction() {
}
#endif
