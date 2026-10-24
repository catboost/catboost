#include "context_config.h"

#if defined(__APPLE__)
    #define CNAME(name) _ ## name
#else
    #define CNAME(name) name
#endif

#define EXPORT(name, id) \
    .globl CNAME(name); \
    .set CNAME(name), id ## f; \
    id:

.text
.p2align 4
.intel_syntax noprefix

.code64

#define MJB_RBX 0
#define MJB_RBP 1
#define MJB_R12 2
#define MJB_R13 3
#define MJB_R14 4
#define MJB_R15 5
#define MJB_RSP 6
#define MJB_PC 7
#define MJB_SIZE (8 * 8)

#if defined(USE_JUMP_CONT)
.p2align 4
.globl CNAME(ContextTrampoLine)
#if defined(__ELF__)
    .type CNAME(ContextTrampoLine), @function
#endif
CNAME(ContextTrampoLine):
        .cfi_startproc
        .cfi_def_cfa rsp, 0
        // Mark RIP undefined to stop unwinding here (DWARF 6.4.4).
        .cfi_undefined rip
#if defined(__CET__) && (__CET__ & 1)
        endbr64
#endif
        mov rdi, [rsp]
#if !defined(NDEBUG)
        cmp rdi, [rsp + 8]
        jne 1f
#endif
        // Shift RSP to 8 mod 16, provide a return address for unwinding.
        call CNAME(ContextTrampoLineRun)
        // ContextTrampoLineRun should never return, if it did, we have a problem.
        call CNAME(abort)
        ud2

#if !defined(NDEBUG)
1:
        mov rsi, [rsp + 8]
        call CNAME(ContextTrampoLineAbort)
        ud2
#endif
        .cfi_endproc
#if defined(__ELF__)
    .size CNAME(ContextTrampoLine), .-CNAME(ContextTrampoLine)
#endif
#endif

EXPORT(__mylongjmp, 1)
        mov rbx, [rdi + MJB_RBX * 8]
        mov rbp, [rdi + MJB_RBP * 8]
        mov r12, [rdi + MJB_R12 * 8]
        mov r13, [rdi + MJB_R13 * 8]
        mov r14, [rdi + MJB_R14 * 8]
        mov r15, [rdi + MJB_R15 * 8]
        test esi, esi
        mov eax, 1
        cmove esi, eax
        mov eax, esi
        mov rdx, [rdi + MJB_PC * 8]
        mov rsp, [rdi + MJB_RSP * 8]
        jmp rdx

EXPORT(__mysetjmp, 2)
        mov [rdi + MJB_RBX * 8], rbx
        mov [rdi + MJB_RBP * 8], rbp
        mov [rdi + MJB_R12 * 8], r12
        mov [rdi + MJB_R13 * 8], r13
        mov [rdi + MJB_R14 * 8], r14
        mov [rdi + MJB_R15 * 8], r15
        lea rdx, [rsp + 8]
        mov [rdi + MJB_RSP * 8], rdx
        mov rax, [rsp]
        mov [rdi + MJB_PC * 8], rax
        mov eax, 0
        ret

#if defined(__ELF__)
    .section .note.GNU-stack,"",@progbits
#endif
