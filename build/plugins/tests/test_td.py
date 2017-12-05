from build.plugins import td


def test_include_parser():
    text = '''
//===----------------------------------------------------------------------===//
// X86 Instruction Format Definitions.
//

include "X86InstrFormats.td"

include "X86InstrExtension.td"

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Pattern fragments.
//

// X86 specific condition code. These correspond to CondCode in
// X86InstrInfo.h. They must be kept in synch.
def X86_COND_A   : PatLeaf<(i8 0)>;  // alt. COND_NBE
def X86_COND_AE  : PatLeaf<(i8 1)>;  // alt. COND_NC
def X86_COND_B   : PatLeaf<(i8 2)>;  // alt. COND_C
'''
    includes = list(td.TableGenParser.parse_includes(text.split('\n')))
    assert includes == [
        "X86InstrFormats.td",
        "X86InstrExtension.td",
        "llvm/Target/Target.td",
    ]
