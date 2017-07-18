        [bits 32]

        %define MJB_BX   0
        %define MJB_SI   1
        %define MJB_DI   2
        %define MJB_BP   3
        %define MJB_SP   4
        %define MJB_PC   5
        %define MJB_RSP  MJB_SP
        %define MJB_SIZE 24

        %define LINKAGE  4
        %define PCOFF    0
        %define PTR_SIZE 4

        %define PARMS  LINKAGE
        %define JMPBUF PARMS
        %define JBUF   PARMS
        %define VAL    JBUF + PTR_SIZE

EXPORT __mylongjmp
        mov ecx, [esp + JBUF]
        mov eax, [esp + VAL]
        mov edx, [ecx + MJB_PC*4]
        mov ebx, [ecx + MJB_BX*4]
        mov esi, [ecx + MJB_SI*4]
        mov edi, [ecx + MJB_DI*4]
        mov ebp, [ecx + MJB_BP*4]
        mov esp, [ecx + MJB_SP*4]
        jmp edx

EXPORT __mysetjmp
        mov eax, [esp + JMPBUF]
        mov [eax + MJB_BX*4], ebx
        mov [eax + MJB_SI*4], esi
        mov [eax + MJB_DI*4], edi
        lea ecx, [esp + JMPBUF]
        mov [eax + MJB_SP*4], ecx
        mov ecx, [esp + PCOFF]
        mov [eax + MJB_PC*4], ecx
        mov [eax + MJB_BP*4], ebp
        xor eax, eax
        ret
