/* ANSI-C code produced by genperf */

#include <util.h>

#include <ctype.h>
#include <libyasm.h>
#include <libyasm/phash.h>

#include "modules/arch/x86/x86arch.h"

#define PROC_8086	0
#define PROC_186	1
#define PROC_286	2
#define PROC_386	3
#define PROC_486	4
#define PROC_586	5
#define PROC_686	6
#define PROC_p2		7
#define PROC_p3		8
#define PROC_p4		9
#define PROC_prescott	10
#define PROC_conroe	11
#define PROC_penryn	12
#define PROC_nehalem	13
#define PROC_westmere	14
#define PROC_sandybridge	15
#define PROC_ivybridge	16
#define PROC_haswell	17
#define PROC_broadwell	18
#define PROC_skylake	19

static void
x86_cpu_intel(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    BitVector_Empty(cpu);

    BitVector_Bit_On(cpu, CPU_Priv);
    if (data >= PROC_286)
        BitVector_Bit_On(cpu, CPU_Prot);
    if (data >= PROC_386)
        BitVector_Bit_On(cpu, CPU_SMM);
    if (data >= PROC_skylake) {
        BitVector_Bit_On(cpu, CPU_SHA);
    }
    if (data >= PROC_broadwell) {
        BitVector_Bit_On(cpu, CPU_RDSEED);
        BitVector_Bit_On(cpu, CPU_ADX);
        BitVector_Bit_On(cpu, CPU_PRFCHW);
    }
    if (data >= PROC_haswell) {
        BitVector_Bit_On(cpu, CPU_FMA);
        BitVector_Bit_On(cpu, CPU_AVX2);
        BitVector_Bit_On(cpu, CPU_BMI1);
        BitVector_Bit_On(cpu, CPU_BMI2);
        BitVector_Bit_On(cpu, CPU_INVPCID);
        BitVector_Bit_On(cpu, CPU_LZCNT);
        BitVector_Bit_On(cpu, CPU_TSX);
        BitVector_Bit_On(cpu, CPU_SMAP);
    }
    if (data >= PROC_ivybridge) {
        BitVector_Bit_On(cpu, CPU_F16C);
        BitVector_Bit_On(cpu, CPU_FSGSBASE);
        BitVector_Bit_On(cpu, CPU_RDRAND);
    }
    if (data >= PROC_sandybridge) {
        BitVector_Bit_On(cpu, CPU_AVX);
        BitVector_Bit_On(cpu, CPU_XSAVEOPT);
        BitVector_Bit_On(cpu, CPU_EPTVPID);
        BitVector_Bit_On(cpu, CPU_SMX);
    }
    if (data >= PROC_westmere) {
        BitVector_Bit_On(cpu, CPU_AES);
        BitVector_Bit_On(cpu, CPU_CLMUL);
    }
    if (data >= PROC_nehalem) {
        BitVector_Bit_On(cpu, CPU_SSE42);
        BitVector_Bit_On(cpu, CPU_XSAVE);
    }
    if (data >= PROC_penryn)
        BitVector_Bit_On(cpu, CPU_SSE41);
    if (data >= PROC_conroe)
        BitVector_Bit_On(cpu, CPU_SSSE3);
    if (data >= PROC_prescott)
        BitVector_Bit_On(cpu, CPU_SSE3);
    if (data >= PROC_p4)
        BitVector_Bit_On(cpu, CPU_SSE2);
    if (data >= PROC_p3)
        BitVector_Bit_On(cpu, CPU_SSE);
    if (data >= PROC_p2)
        BitVector_Bit_On(cpu, CPU_MMX);
    if (data >= PROC_486)
        BitVector_Bit_On(cpu, CPU_FPU);
    if (data >= PROC_prescott)
        BitVector_Bit_On(cpu, CPU_EM64T);

    if (data >= PROC_p4)
        BitVector_Bit_On(cpu, CPU_P4);
    if (data >= PROC_p3)
        BitVector_Bit_On(cpu, CPU_P3);
    if (data >= PROC_686)
        BitVector_Bit_On(cpu, CPU_686);
    if (data >= PROC_586)
        BitVector_Bit_On(cpu, CPU_586);
    if (data >= PROC_486)
        BitVector_Bit_On(cpu, CPU_486);
    if (data >= PROC_386)
        BitVector_Bit_On(cpu, CPU_386);
    if (data >= PROC_286)
        BitVector_Bit_On(cpu, CPU_286);
    if (data >= PROC_186)
        BitVector_Bit_On(cpu, CPU_186);
    BitVector_Bit_On(cpu, CPU_086);

    /* Use Intel long NOPs if 686 or better */
    if (data >= PROC_686)
        arch_x86->nop = X86_NOP_INTEL;
    else
        arch_x86->nop = X86_NOP_BASIC;
}

static void
x86_cpu_ia64(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    BitVector_Empty(cpu);
    BitVector_Bit_On(cpu, CPU_Priv);
    BitVector_Bit_On(cpu, CPU_Prot);
    BitVector_Bit_On(cpu, CPU_SMM);
    BitVector_Bit_On(cpu, CPU_SSE2);
    BitVector_Bit_On(cpu, CPU_SSE);
    BitVector_Bit_On(cpu, CPU_MMX);
    BitVector_Bit_On(cpu, CPU_FPU);
    BitVector_Bit_On(cpu, CPU_IA64);
    BitVector_Bit_On(cpu, CPU_P4);
    BitVector_Bit_On(cpu, CPU_P3);
    BitVector_Bit_On(cpu, CPU_686);
    BitVector_Bit_On(cpu, CPU_586);
    BitVector_Bit_On(cpu, CPU_486);
    BitVector_Bit_On(cpu, CPU_386);
    BitVector_Bit_On(cpu, CPU_286);
    BitVector_Bit_On(cpu, CPU_186);
    BitVector_Bit_On(cpu, CPU_086);
}

#define PROC_bulldozer	11
#define PROC_k10    10
#define PROC_venice 9
#define PROC_hammer 8
#define PROC_k7     7
#define PROC_k6     6

static void
x86_cpu_amd(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    BitVector_Empty(cpu);

    BitVector_Bit_On(cpu, CPU_Priv);
    BitVector_Bit_On(cpu, CPU_Prot);
    BitVector_Bit_On(cpu, CPU_SMM);
    BitVector_Bit_On(cpu, CPU_3DNow);
    if (data >= PROC_bulldozer) {
        BitVector_Bit_On(cpu, CPU_XOP);
        BitVector_Bit_On(cpu, CPU_FMA4);
    }
    if (data >= PROC_k10)
        BitVector_Bit_On(cpu, CPU_SSE4a);
    if (data >= PROC_venice)
        BitVector_Bit_On(cpu, CPU_SSE3);
    if (data >= PROC_hammer)
        BitVector_Bit_On(cpu, CPU_SSE2);
    if (data >= PROC_k7)
        BitVector_Bit_On(cpu, CPU_SSE);
    if (data >= PROC_k6)
        BitVector_Bit_On(cpu, CPU_MMX);
    BitVector_Bit_On(cpu, CPU_FPU);

    if (data >= PROC_hammer)
        BitVector_Bit_On(cpu, CPU_Hammer);
    if (data >= PROC_k7)
        BitVector_Bit_On(cpu, CPU_Athlon);
    if (data >= PROC_k6)
        BitVector_Bit_On(cpu, CPU_K6);
    BitVector_Bit_On(cpu, CPU_686);
    BitVector_Bit_On(cpu, CPU_586);
    BitVector_Bit_On(cpu, CPU_486);
    BitVector_Bit_On(cpu, CPU_386);
    BitVector_Bit_On(cpu, CPU_286);
    BitVector_Bit_On(cpu, CPU_186);
    BitVector_Bit_On(cpu, CPU_086);

    /* Use AMD long NOPs if k6 or better */
    if (data >= PROC_k6)
        arch_x86->nop = X86_NOP_AMD;
    else
        arch_x86->nop = X86_NOP_BASIC;
}

static void
x86_cpu_set(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    BitVector_Bit_On(cpu, data);
}

static void
x86_cpu_clear(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    BitVector_Bit_Off(cpu, data);
}

static void
x86_cpu_set_sse4(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    BitVector_Bit_On(cpu, CPU_SSE41);
    BitVector_Bit_On(cpu, CPU_SSE42);
}

static void
x86_cpu_clear_sse4(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    BitVector_Bit_Off(cpu, CPU_SSE41);
    BitVector_Bit_Off(cpu, CPU_SSE42);
}

static void
x86_nop(wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data)
{
    arch_x86->nop = data;
}

struct cpu_parse_data {
    const char *name;
    void (*handler) (wordptr cpu, yasm_arch_x86 *arch_x86, unsigned int data);
    unsigned int data;
};
static const struct cpu_parse_data *
cpu_find(const char *key, size_t len)
{
  static const struct cpu_parse_data pd[179] = {
    {"noeptvpid",	x86_cpu_clear,	CPU_EPTVPID},
    {"amd",		x86_cpu_set,	CPU_AMD},
    {"sse41",		x86_cpu_set,	CPU_SSE41},
    {"pentium",	x86_cpu_intel,	PROC_586},
    {"intelnop",	x86_nop,	X86_NOP_INTEL},
    {"pclmulqdq",	x86_cpu_set,	CPU_CLMUL},
    {"sse42",		x86_cpu_set,	CPU_SSE42},
    {"nobmi2",		x86_cpu_clear,	CPU_BMI2},
    {"pentium3",	x86_cpu_intel,	PROC_p3},
    {"broadwell",	x86_cpu_intel,	PROC_broadwell},
    {"aes",		x86_cpu_set,	CPU_AES},
    {"eptvpid",	x86_cpu_set,	CPU_EPTVPID},
    {"f16c",		x86_cpu_set,	CPU_F16C},
    {"amdnop",		x86_nop,	X86_NOP_AMD},
    {"pentium-2",	x86_cpu_intel,	PROC_p2},
    {"nofpu",		x86_cpu_clear,	CPU_FPU},
    {"bmi2",		x86_cpu_set,	CPU_BMI2},
    {"katmai",		x86_cpu_intel,	PROC_p3},
    {"pentiumiii",	x86_cpu_intel,	PROC_p3},
    {"fpu",		x86_cpu_set,	CPU_FPU},
    {"noundoc",	x86_cpu_clear,	CPU_Undoc},
    {"no3dnow",	x86_cpu_clear,	CPU_3DNow},
    {"i486",		x86_cpu_intel,	PROC_486},
    {"noundocumented",	x86_cpu_clear,	CPU_Undoc},
    {"sse",		x86_cpu_set,	CPU_SSE},
    {"nossse3",	x86_cpu_clear,	CPU_SSSE3},
    {"noclmul",	x86_cpu_clear,	CPU_CLMUL},
    {"tsx",		x86_cpu_set,	CPU_TSX},
    {"nocyrix",	x86_cpu_clear,	CPU_Cyrix},
    {"nosse",		x86_cpu_clear,	CPU_SSE},
    {"nofma",		x86_cpu_clear,	CPU_FMA},
    {"phenom",		x86_cpu_amd,	PROC_k10},
    {"haswell",	x86_cpu_intel,	PROC_haswell},
    {"noprot",		x86_cpu_clear,	CPU_Prot},
    {"padlock",	x86_cpu_set,	CPU_PadLock},
    {"nopclmulqdq",	x86_cpu_clear,	CPU_CLMUL},
    {"nofma4",		x86_cpu_clear,	CPU_FMA4},
    {"nofsgsbase",	x86_cpu_clear,	CPU_FSGSBASE},
    {"prot",		x86_cpu_set,	CPU_Prot},
    {"opteron",	x86_cpu_amd,	PROC_hammer},
    {"nof16c",		x86_cpu_clear,	CPU_F16C},
    {"i386",		x86_cpu_intel,	PROC_386},
    {"ssse3",		x86_cpu_set,	CPU_SSSE3},
    {"protected",	x86_cpu_set,	CPU_Prot},
    {"bulldozer",	x86_cpu_amd,	PROC_bulldozer},
    {"lzcnt",		x86_cpu_set,	CPU_LZCNT},
    {"obs",		x86_cpu_set,	CPU_Obs},
    {"noprotected",	x86_cpu_clear,	CPU_Prot},
    {"athlon-64",	x86_cpu_amd,	PROC_hammer},
    {"undocumented",	x86_cpu_set,	CPU_Undoc},
    {"i686",		x86_cpu_intel,	PROC_686},
    {"k8",		x86_cpu_amd,	PROC_hammer},
    {"k10",		x86_cpu_amd,	PROC_k10},
    {"noavx2",		x86_cpu_clear,	CPU_AVX2},
    {"sandybridge",	x86_cpu_intel,	PROC_sandybridge},
    {"nommx",		x86_cpu_clear,	CPU_MMX},
    {"priv",		x86_cpu_set,	CPU_Priv},
    {"sse4.1",		x86_cpu_set,	CPU_SSE41},
    {"8086",		x86_cpu_intel,	PROC_8086},
    {"noprivileged",	x86_cpu_clear,	CPU_Priv},
    {"i586",		x86_cpu_intel,	PROC_586},
    {"ia-64",		x86_cpu_ia64,	0},
    {"nosse2",		x86_cpu_clear,	CPU_SSE2},
    {"obsolete",	x86_cpu_set,	CPU_Obs},
    {"186",		x86_cpu_intel,	PROC_186},
    {"sse4a",		x86_cpu_set,	CPU_SSE4a},
    {"ia64",		x86_cpu_ia64,	0},
    {"core2",		x86_cpu_intel,	PROC_conroe},
    {"noxsaveopt",	x86_cpu_clear,	CPU_XSAVEOPT},
    {"sse4.2",		x86_cpu_set,	CPU_SSE42},
    {"prescott",	x86_cpu_intel,	PROC_prescott},
    {"avx2",		x86_cpu_set,	CPU_AVX2},
    {"80186",		x86_cpu_intel,	PROC_186},
    {"nopriv",		x86_cpu_clear,	CPU_Priv},
    {"nosse4.1",	x86_cpu_clear,	CPU_SSE41},
    {"nordseed",	x86_cpu_clear,	CPU_RDSEED},
    {"pentium2",	x86_cpu_intel,	PROC_p2},
    {"conroe",		x86_cpu_intel,	PROC_conroe},
    {"nosse42",	x86_cpu_clear,	CPU_SSE42},
    {"pentium-ii",	x86_cpu_intel,	PROC_p2},
    {"svm",		x86_cpu_set,	CPU_SVM},
    {"386",		x86_cpu_intel,	PROC_386},
    {"em64t",		x86_cpu_set,	CPU_EM64T},
    {"p2",		x86_cpu_intel,	PROC_p2},
    {"athlon64",	x86_cpu_amd,	PROC_hammer},
    {"3dnow",		x86_cpu_set,	CPU_3DNow},
    {"nosse4",		x86_cpu_clear_sse4,	0},
    {"nosmx",		x86_cpu_clear,	CPU_SMX},
    {"williamette",	x86_cpu_intel,	PROC_p4},
    {"family10h",	x86_cpu_amd,	PROC_k10},
    {"athlon",		x86_cpu_amd,	PROC_k7},
    {"586",		x86_cpu_intel,	PROC_586},
    {"686",		x86_cpu_intel,	PROC_686},
    {"smm",		x86_cpu_set,	CPU_SMM},
    {"xsave",		x86_cpu_set,	CPU_XSAVE},
    {"privileged",	x86_cpu_set,	CPU_Priv},
    {"p6",		x86_cpu_intel,	PROC_686},
    {"smap",		x86_cpu_set,	CPU_SMAP},
    {"avx",		x86_cpu_set,	CPU_AVX},
    {"pentium-4",	x86_cpu_intel,	PROC_p4},
    {"pentiumii",	x86_cpu_intel,	PROC_p2},
    {"sha",		x86_cpu_set,	CPU_SHA},
    {"fma4",		x86_cpu_set,	CPU_FMA4},
    {"pentium-iii",	x86_cpu_intel,	PROC_p3},
    {"skylake",	x86_cpu_intel,	PROC_skylake},
    {"nosse4.2",	x86_cpu_clear,	CPU_SSE42},
    {"pentium4",	x86_cpu_intel,	PROC_p4},
    {"noaes",		x86_cpu_clear,	CPU_AES},
    {"i186",		x86_cpu_intel,	PROC_186},
    {"rdrand",		x86_cpu_set,	CPU_RDRAND},
    {"80286",		x86_cpu_intel,	PROC_286},
    {"pentiumiv",	x86_cpu_intel,	PROC_p4},
    {"xop",		x86_cpu_set,	CPU_XOP},
    {"mmx",		x86_cpu_set,	CPU_MMX},
    {"486",		x86_cpu_intel,	PROC_486},
    {"clawhammer",	x86_cpu_amd,	PROC_hammer},
    {"rdseed",		x86_cpu_set,	CPU_RDSEED},
    {"i286",		x86_cpu_intel,	PROC_286},
    {"prfchw",		x86_cpu_set,	CPU_PRFCHW},
    {"nosse3",		x86_cpu_clear,	CPU_SSE3},
    {"sse4",		x86_cpu_set_sse4,	0},
    {"pentium-iv",	x86_cpu_intel,	PROC_p4},
    {"p4",		x86_cpu_intel,	PROC_p4},
    {"nordrand",	x86_cpu_clear,	CPU_RDRAND},
    {"ppro",		x86_cpu_intel,	PROC_686},
    {"p5",		x86_cpu_intel,	PROC_586},
    {"notbm",	x86_cpu_clear,	CPU_TBM},
    {"cyrix",		x86_cpu_set,	CPU_Cyrix},
    {"80386",		x86_cpu_intel,	PROC_386},
    {"k6",		x86_cpu_amd,	PROC_k6},
    {"basicnop",	x86_nop,	X86_NOP_BASIC},
    {"nomovbe",	x86_cpu_clear,	CPU_MOVBE},
    {"noadx",		x86_cpu_clear,	CPU_ADX},
    {"nosmap",		x86_cpu_clear,	CPU_SMAP},
    {"nosmm",		x86_cpu_clear,	CPU_SMM},
    {"xsaveopt",	x86_cpu_set,	CPU_XSAVEOPT},
    {"pentium-3",	x86_cpu_intel,	PROC_p3},
    {"nosvm",		x86_cpu_clear,	CPU_SVM},
    {"nosha",		x86_cpu_clear,	CPU_SHA},
    {"invpcid",	x86_cpu_set,	CPU_INVPCID},
    {"nobmi1",		x86_cpu_clear,	CPU_BMI1},
    {"ivybridge",	x86_cpu_intel,	PROC_ivybridge},
    {"p3",		x86_cpu_intel,	PROC_p3},
    {"pentiumpro",	x86_cpu_intel,	PROC_686},
    {"penryn",		x86_cpu_intel,	PROC_penryn},
    {"80486",		x86_cpu_intel,	PROC_486},
    {"noxop",		x86_cpu_clear,	CPU_XOP},
    {"undoc",		x86_cpu_set,	CPU_Undoc},
    {"noobsolete",	x86_cpu_clear,	CPU_Obs},
    {"noavx",		x86_cpu_clear,	CPU_AVX},
    {"nolzcnt",	x86_cpu_clear,	CPU_LZCNT},
    {"noprfchw",	x86_cpu_clear,	CPU_PRFCHW},
    {"notsx",		x86_cpu_clear,	CPU_TSX},
    {"bmi1",		x86_cpu_set,	CPU_BMI1},
    {"itanium",	x86_cpu_ia64,	0},
    {"venice",		x86_cpu_amd,	PROC_venice},
    {"noxsave",	x86_cpu_clear,	CPU_XSAVE},
    {"noamd",		x86_cpu_clear,	CPU_AMD},
    {"noobs",		x86_cpu_clear,	CPU_Obs},
    {"noem64t",	x86_cpu_clear,	CPU_EM64T},
    {"hammer",		x86_cpu_amd,	PROC_hammer},
    {"nehalem",	x86_cpu_intel,	PROC_nehalem},
    {"sse3",		x86_cpu_set,	CPU_SSE3},
    {"sse2",		x86_cpu_set,	CPU_SSE2},
    {"clmul",		x86_cpu_set,	CPU_CLMUL},
    {"smx",		x86_cpu_set,	CPU_SMX},
    {"nosse4a",	x86_cpu_clear,	CPU_SSE4a},
    {"tbm",		x86_cpu_set,	CPU_TBM},
    {"fma",		x86_cpu_set,	CPU_FMA},
    {"nopadlock",	x86_cpu_clear,	CPU_PadLock},
    {"nosse41",	x86_cpu_clear,	CPU_SSE41},
    {"adx",		x86_cpu_set,	CPU_ADX},
    {"westmere",	x86_cpu_intel,	PROC_westmere},
    {"k7",		x86_cpu_amd,	PROC_k7},
    {"noinvpcid",	x86_cpu_clear,	CPU_INVPCID},
    {"fsgsbase",	x86_cpu_set,	CPU_FSGSBASE},
    {"corei7",		x86_cpu_intel,	PROC_nehalem},
    {"movbe",		x86_cpu_set,	CPU_MOVBE},
    {"286",		x86_cpu_intel,	PROC_286}
  };
  static const unsigned char tab[] = {
    183,125,113,40,125,0,0,0,183,146,116,85,0,113,113,183,
    113,131,0,82,88,0,131,125,85,0,113,0,0,7,0,40,
    22,7,0,0,125,220,87,183,184,7,0,0,0,113,11,0,
    84,0,0,0,0,131,0,113,0,120,0,113,0,0,51,11,
    55,190,0,0,183,61,120,131,85,135,0,0,0,0,0,82,
    74,183,0,87,220,0,235,0,220,229,0,0,220,243,124,145,
    0,220,131,0,221,0,0,0,237,0,135,125,124,168,0,69,
    0,124,22,0,131,131,163,113,184,214,155,133,55,0,0,0,
  };

  const struct cpu_parse_data *ret;
  unsigned long rsl, val = phash_lookup(key, len, 0xdaa66d2bUL);
  rsl = ((val>>25)^tab[val&0x7f]);
  if (rsl >= 179) return NULL;
  ret = &pd[rsl];
  if (strcmp(key, ret->name) != 0) return NULL;
  return ret;
}



void
yasm_x86__parse_cpu(yasm_arch_x86 *arch_x86, const char *cpuid,
                    size_t cpuid_len)
{
    /*@null@*/ const struct cpu_parse_data *pdata;
    wordptr new_cpu;
    size_t i;
    static char lcaseid[16];

    if (cpuid_len > 15)
        return;
    for (i=0; i<cpuid_len; i++)
        lcaseid[i] = tolower(cpuid[i]);
    lcaseid[cpuid_len] = '\0';

    pdata = cpu_find(lcaseid, cpuid_len);
    if (!pdata) {
        yasm_warn_set(YASM_WARN_GENERAL,
                      N_("unrecognized CPU identifier `%s'"), cpuid);
        return;
    }

    new_cpu = BitVector_Clone(arch_x86->cpu_enables[arch_x86->active_cpu]);
    pdata->handler(new_cpu, arch_x86, pdata->data);

    /* try to find an existing match in the CPU table first */
    for (i=0; i<arch_x86->cpu_enables_size; i++) {
        if (BitVector_equal(arch_x86->cpu_enables[i], new_cpu)) {
            arch_x86->active_cpu = i;
            BitVector_Destroy(new_cpu);
            return;
        }
    }

    /* not found, need to add a new entry */
    arch_x86->active_cpu = arch_x86->cpu_enables_size++;
    arch_x86->cpu_enables =
        yasm_xrealloc(arch_x86->cpu_enables,
                      arch_x86->cpu_enables_size*sizeof(wordptr));
    arch_x86->cpu_enables[arch_x86->active_cpu] = new_cpu;
}

