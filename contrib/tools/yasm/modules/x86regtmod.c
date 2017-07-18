/* ANSI-C code produced by genperf */

#include <util.h>

#include <ctype.h>
#include <libyasm.h>
#include <libyasm/phash.h>

#include "modules/arch/x86/x86arch.h"

enum regtmod_type {
    REG = 1,
    REGGROUP,
    SEGREG,
    TARGETMOD
};
struct regtmod_parse_data {
    const char *name;
    unsigned int type:8;                /* regtmod_type */

    /* REG: register size
     * SEGREG: prefix encoding
     * Others: 0
     */
    unsigned int size_prefix:8;

    /* REG: register index
     * REGGROUP: register group type
     * SEGREG: register encoding
     * TARGETMOD: target modifier
     */
    unsigned int data:8;

    /* REG: required bits setting
     * SEGREG: BITS in which the segment is ignored
     * Others: 0
     */
    unsigned int bits:8;
};
static const struct regtmod_parse_data *
regtmod_find(const char *key, size_t len)
{
  static const struct regtmod_parse_data pd[152] = {
    {"st6",	REG,	X86_FPUREG,	6,	0},
    {"ymm1",	REG,	X86_YMMREG,	1,	0},
    {"rsp",	REG,	X86_REG64,	4,	64},
    {"r10",	REG,	X86_REG64,	10,	64},
    {"bp",	REG,	X86_REG16,	5,	0},
    {"r10b",	REG,	X86_REG8,	10,	64},
    {"mm2",	REG,	X86_MMXREG,	2,	0},
    {"rdi",	REG,	X86_REG64,	7,	64},
    {"ymm12",	REG,	X86_YMMREG,	12,	64},
    {"r8",	REG,	X86_REG64,	8,	64},
    {"gs",	SEGREG,	0x65,	0x05,	0},
    {"r10d",	REG,	X86_REG32,	10,	64},
    {"rsi",	REG,	X86_REG64,	6,	64},
    {"eax",	REG,	X86_REG32,	0,	0},
    {"mm3",	REG,	X86_MMXREG,	3,	0},
    {"tr6",	REG,	X86_TRREG,	6,	0},
    {"tr0",	REG,	X86_TRREG,	0,	0},
    {"ah",	REG,	X86_REG8,	4,	0},
    {"xmm6",	REG,	X86_XMMREG,	6,	0},
    {"dr6",	REG,	X86_DRREG,	6,	0},
    {"esp",	REG,	X86_REG32,	4,	0},
    {"bpl",	REG,	X86_REG8X,	5,	64},
    {"tr5",	REG,	X86_TRREG,	5,	0},
    {"ax",	REG,	X86_REG16,	0,	0},
    {"sp",	REG,	X86_REG16,	4,	0},
    {"r15b",	REG,	X86_REG8,	15,	64},
    {"xmm14",	REG,	X86_XMMREG,	14,	64},
    {"xmm12",	REG,	X86_XMMREG,	12,	64},
    {"r11",	REG,	X86_REG64,	11,	64},
    {"xmm",	REGGROUP,	0,	X86_XMMREG,	0},
    {"ymm2",	REG,	X86_YMMREG,	2,	0},
    {"ebp",	REG,	X86_REG32,	5,	0},
    {"xmm8",	REG,	X86_XMMREG,	8,	64},
    {"r12d",	REG,	X86_REG32,	12,	64},
    {"ymm4",	REG,	X86_YMMREG,	4,	0},
    {"ymm3",	REG,	X86_YMMREG,	3,	0},
    {"rax",	REG,	X86_REG64,	0,	64},
    {"xmm3",	REG,	X86_XMMREG,	3,	0},
    {"xmm0",	REG,	X86_XMMREG,	0,	0},
    {"dr7",	REG,	X86_DRREG,	7,	0},
    {"r14w",	REG,	X86_REG16,	14,	64},
    {"mm1",	REG,	X86_MMXREG,	1,	0},
    {"bl",	REG,	X86_REG8,	3,	0},
    {"r9w",	REG,	X86_REG16,	9,	64},
    {"ymm13",	REG,	X86_YMMREG,	13,	64},
    {"r9b",	REG,	X86_REG8,	9,	64},
    {"ymm8",	REG,	X86_YMMREG,	8,	64},
    {"dx",	REG,	X86_REG16,	2,	0},
    {"r12",	REG,	X86_REG64,	12,	64},
    {"r12w",	REG,	X86_REG16,	12,	64},
    {"r9",	REG,	X86_REG64,	9,	64},
    {"r15w",	REG,	X86_REG16,	15,	64},
    {"sil",	REG,	X86_REG8X,	6,	64},
    {"r10w",	REG,	X86_REG16,	10,	64},
    {"ymm6",	REG,	X86_YMMREG,	6,	0},
    {"ss",	SEGREG,	0x36,	0x02,	64},
    {"tr4",	REG,	X86_TRREG,	4,	0},
    {"cr3",	REG,	X86_CRREG,	3,	0},
    {"r11w",	REG,	X86_REG16,	11,	64},
    {"xmm4",	REG,	X86_XMMREG,	4,	0},
    {"st0",	REG,	X86_FPUREG,	0,	0},
    {"dil",	REG,	X86_REG8X,	7,	64},
    {"tr3",	REG,	X86_TRREG,	3,	0},
    {"r13d",	REG,	X86_REG32,	13,	64},
    {"r8w",	REG,	X86_REG16,	8,	64},
    {"xmm13",	REG,	X86_XMMREG,	13,	64},
    {"st3",	REG,	X86_FPUREG,	3,	0},
    {"xmm15",	REG,	X86_XMMREG,	15,	64},
    {"xmm10",	REG,	X86_XMMREG,	10,	64},
    {"es",	SEGREG,	0x26,	0x00,	64},
    {"cr8",	REG,	X86_CRREG,	8,	64},
    {"xmm7",	REG,	X86_XMMREG,	7,	0},
    {"spl",	REG,	X86_REG8X,	4,	64},
    {"r15",	REG,	X86_REG64,	15,	64},
    {"cr4",	REG,	X86_CRREG,	4,	0},
    {"fs",	SEGREG,	0x64,	0x04,	0},
    {"rcx",	REG,	X86_REG64,	1,	64},
    {"mm0",	REG,	X86_MMXREG,	0,	0},
    {"mm",	REGGROUP,	0,	X86_MMXREG,	0},
    {"tr1",	REG,	X86_TRREG,	1,	0},
    {"short",	TARGETMOD,	0,	X86_SHORT,	0},
    {"st4",	REG,	X86_FPUREG,	4,	0},
    {"cr0",	REG,	X86_CRREG,	0,	0},
    {"xmm11",	REG,	X86_XMMREG,	11,	64},
    {"mm4",	REG,	X86_MMXREG,	4,	0},
    {"bh",	REG,	X86_REG8,	7,	0},
    {"r15d",	REG,	X86_REG32,	15,	64},
    {"dr2",	REG,	X86_DRREG,	2,	0},
    {"r8d",	REG,	X86_REG32,	8,	64},
    {"ymm7",	REG,	X86_YMMREG,	7,	0},
    {"xmm9",	REG,	X86_XMMREG,	9,	64},
    {"r12b",	REG,	X86_REG8,	12,	64},
    {"st2",	REG,	X86_FPUREG,	2,	0},
    {"ymm9",	REG,	X86_YMMREG,	9,	64},
    {"st7",	REG,	X86_FPUREG,	7,	0},
    {"bx",	REG,	X86_REG16,	3,	0},
    {"ymm11",	REG,	X86_YMMREG,	11,	64},
    {"ymm5",	REG,	X86_YMMREG,	5,	0},
    {"ymm15",	REG,	X86_YMMREG,	15,	64},
    {"rbp",	REG,	X86_REG64,	5,	64},
    {"r13",	REG,	X86_REG64,	13,	64},
    {"mm5",	REG,	X86_MMXREG,	5,	0},
    {"si",	REG,	X86_REG16,	6,	0},
    {"dl",	REG,	X86_REG8,	2,	0},
    {"di",	REG,	X86_REG16,	7,	0},
    {"cr2",	REG,	X86_CRREG,	2,	0},
    {"r14d",	REG,	X86_REG32,	14,	64},
    {"ymm14",	REG,	X86_YMMREG,	14,	64},
    {"tr7",	REG,	X86_TRREG,	7,	0},
    {"ds",	SEGREG,	0x3e,	0x03,	64},
    {"cx",	REG,	X86_REG16,	1,	0},
    {"st",	REGGROUP,	0,	X86_FPUREG,	0},
    {"edi",	REG,	X86_REG32,	7,	0},
    {"al",	REG,	X86_REG8,	0,	0},
    {"mm7",	REG,	X86_MMXREG,	7,	0},
    {"ebx",	REG,	X86_REG32,	3,	0},
    {"xmm2",	REG,	X86_XMMREG,	2,	0},
    {"st5",	REG,	X86_FPUREG,	5,	0},
    {"rip",	REG,	X86_RIP,	0,	64},
    {"rbx",	REG,	X86_REG64,	3,	64},
    {"to",	TARGETMOD,	0,	X86_TO,		0},
    {"r11d",	REG,	X86_REG32,	11,	64},
    {"dr5",	REG,	X86_DRREG,	5,	0},
    {"dr1",	REG,	X86_DRREG,	1,	0},
    {"near",	TARGETMOD,	0,	X86_NEAR,	0},
    {"r14",	REG,	X86_REG64,	14,	64},
    {"dh",	REG,	X86_REG8,	6,	0},
    {"cl",	REG,	X86_REG8,	1,	0},
    {"dr4",	REG,	X86_DRREG,	4,	0},
    {"ymm10",	REG,	X86_YMMREG,	10,	64},
    {"dr3",	REG,	X86_DRREG,	3,	0},
    {"xmm5",	REG,	X86_XMMREG,	5,	0},
    {"mm6",	REG,	X86_MMXREG,	6,	0},
    {"r13w",	REG,	X86_REG16,	13,	64},
    {"far",	TARGETMOD,	0,	X86_FAR,	0},
    {"ymm0",	REG,	X86_YMMREG,	0,	0},
    {"ymm",	REGGROUP,	0,	X86_YMMREG,	0},
    {"ch",	REG,	X86_REG8,	5,	0},
    {"xmm1",	REG,	X86_XMMREG,	1,	0},
    {"esi",	REG,	X86_REG32,	6,	0},
    {"r14b",	REG,	X86_REG8,	14,	64},
    {"st1",	REG,	X86_FPUREG,	1,	0},
    {"r8b",	REG,	X86_REG8,	8,	64},
    {"r11b",	REG,	X86_REG8,	11,	64},
    {"r13b",	REG,	X86_REG8,	13,	64},
    {"ecx",	REG,	X86_REG32,	1,	0},
    {"tr2",	REG,	X86_TRREG,	2,	0},
    {"rdx",	REG,	X86_REG64,	2,	64},
    {"r9d",	REG,	X86_REG32,	9,	64},
    {"cs",	SEGREG,	0x2e,	0x01,	0},
    {"dr0",	REG,	X86_DRREG,	0,	0},
    {"edx",	REG,	X86_REG32,	2,	0}
  };
  static const unsigned char tab[] = {
    0,0,125,22,0,0,85,0,85,168,0,0,0,7,113,0,
    0,0,0,22,183,0,0,11,42,55,0,0,82,0,88,235,
    0,0,0,0,0,0,183,85,0,0,145,113,220,125,22,0,
    88,183,0,7,0,0,0,7,0,125,113,87,131,116,7,0,
    113,7,0,113,0,87,87,7,7,7,113,40,85,125,113,85,
    0,0,22,235,0,131,125,113,0,22,0,220,0,220,0,120,
    116,0,124,184,0,0,0,183,92,125,0,92,125,0,0,177,
    7,0,0,7,0,45,0,214,180,113,211,163,142,0,88,173,
  };

  const struct regtmod_parse_data *ret;
  unsigned long rsl, val = phash_lookup(key, len, 0x9e3779b9UL);
  rsl = ((val>>25)^tab[val&0x7f]);
  if (rsl >= 152) return NULL;
  ret = &pd[rsl];
  if (strcmp(key, ret->name) != 0) return NULL;
  return ret;
}



yasm_arch_regtmod
yasm_x86__parse_check_regtmod(yasm_arch *arch, const char *id, size_t id_len,
                              uintptr_t *data)
{
    yasm_arch_x86 *arch_x86 = (yasm_arch_x86 *)arch;
    /*@null@*/ const struct regtmod_parse_data *pdata;
    size_t i;
    static char lcaseid[8];
    unsigned int bits;
    yasm_arch_regtmod type;

    if (id_len > 7)
        return YASM_ARCH_NOTREGTMOD;
    for (i=0; i<id_len; i++)
        lcaseid[i] = tolower(id[i]);
    lcaseid[id_len] = '\0';

    pdata = regtmod_find(lcaseid, id_len);
    if (!pdata)
        return YASM_ARCH_NOTREGTMOD;

    type = (yasm_arch_regtmod)pdata->type;
    bits = pdata->bits;

    if (type == YASM_ARCH_REG && bits != 0 && arch_x86->mode_bits != bits) {
        yasm_warn_set(YASM_WARN_GENERAL,
                      N_("`%s' is a register in %u-bit mode"), id, bits);
        return YASM_ARCH_NOTREGTMOD;
    }

    if (type == YASM_ARCH_SEGREG && bits != 0 && arch_x86->mode_bits == bits) {
        yasm_warn_set(YASM_WARN_GENERAL,
                      N_("`%s' segment register ignored in %u-bit mode"), id,
                      bits);
    }

    if (type == YASM_ARCH_SEGREG)
        *data = (pdata->size_prefix<<8) | pdata->data;
    else
        *data = pdata->size_prefix | pdata->data;
    return type;
}

