%macro EXPORT 1
    %ifdef DARWIN
        global _%1
        _%1:
    %else
        global %1
        %1:
    %endif
%endmacro

%ifdef _x86_64_
    %include "context_x86_64.asm"
%else
    %include "context_i686.asm"
%endif
