from build.plugins import yasm


def test_include_parser_1():
    swg_file = """
%macro EXPORT 1
    %ifdef DARWIN
        global _%1
        _%1:
    %else
        global %1
        %1:
    %endif
%endmacro

%include "include1.asm"
%INCLUDE "include2.asm"

%ifdef _x86_64_
    %include "context_x86_64.asm"
%else
    %include "context_i686.asm"
%endif

; Copyright (c) 2008-2013 GNU General Public License www.gnu.org/licenses
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; structure definition and constants:
%INCLUDE "randomah.asi"

global _MersRandomInit, _MersRandomInitByArray
"""

    includes = yasm.YasmParser.parse_includes(swg_file.split('\n'))
    assert includes == ['include1.asm', 'include2.asm', 'context_x86_64.asm', 'context_i686.asm', 'randomah.asi']
