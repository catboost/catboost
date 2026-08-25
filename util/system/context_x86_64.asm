        [bits 64]

        %define MJB_RBX    0
        %define MJB_RBP    1
        %define MJB_R12    2
        %define MJB_R13    3
        %define MJB_R14    4
        %define MJB_R15    5
        %define MJB_RSP    6
        %define MJB_PC     7
        %define MJB_SIZE (8*8)

EXPORT __mylongjmp
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

EXPORT __mysetjmp
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
