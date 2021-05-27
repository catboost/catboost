

PROGRAM(yasm)

NO_CLANG_COVERAGE()

NO_COMPILER_WARNINGS()

NO_UTIL()

NO_PLATFORM()

ALLOCATOR(FAKE)

ADDINCL(
    contrib/tools/yasm
    contrib/tools/yasm/frontends/yasm
    contrib/tools/yasm/modules
)

CFLAGS(
    -DHAVE_CONFIG_H
    -DYASM_LIB_SOURCE
)

SRCDIR(contrib/tools/yasm)

SRCS(
    frontends/yasm/yasm-options.c
    frontends/yasm/yasm.c
    libyasm/assocdat.c
    libyasm/bc-align.c
    libyasm/bc-data.c
    libyasm/bc-incbin.c
    libyasm/bc-org.c
    libyasm/bc-reserve.c
    libyasm/bitvect.c
    libyasm/bytecode.c
    libyasm/cmake-module.c
    libyasm/errwarn.c
    libyasm/expr.c
    libyasm/file.c
    libyasm/floatnum.c
    libyasm/hamt.c
    libyasm/insn.c
    libyasm/intnum.c
    libyasm/inttree.c
    libyasm/linemap.c
    libyasm/md5.c
    libyasm/mergesort.c
    libyasm/phash.c
    libyasm/replace_path.c
    libyasm/section.c
    libyasm/strcasecmp.c
    libyasm/strsep.c
    libyasm/symrec.c
    libyasm/valparam.c
    libyasm/value.c
    libyasm/xmalloc.c
    libyasm/xstrdup.c
    modules/arch/lc3b/lc3barch.c
    modules/arch/lc3b/lc3bbc.c
    modules/arch/x86/x86arch.c
    modules/arch/x86/x86bc.c
    modules/arch/x86/x86expr.c
    modules/arch/x86/x86id.c
    modules/dbgfmts/codeview/cv-dbgfmt.c
    modules/dbgfmts/codeview/cv-symline.c
    modules/dbgfmts/codeview/cv-type.c
    modules/dbgfmts/dwarf2/dwarf2-aranges.c
    modules/dbgfmts/dwarf2/dwarf2-dbgfmt.c
    modules/dbgfmts/dwarf2/dwarf2-info.c
    modules/dbgfmts/dwarf2/dwarf2-line.c
    modules/dbgfmts/null/null-dbgfmt.c
    modules/dbgfmts/stabs/stabs-dbgfmt.c
    modules/gas-token.c
    modules/init_plugin.c
    modules/lc3bid.c
    modules/listfmts/nasm/nasm-listfmt.c
    modules/nasm-token.c
    modules/objfmts/bin/bin-objfmt.c
    modules/objfmts/coff/coff-objfmt.c
    modules/objfmts/coff/win64-except.c
    modules/objfmts/dbg/dbg-objfmt.c
    modules/objfmts/elf/elf-objfmt.c
    modules/objfmts/elf/elf-x86-amd64.c
    modules/objfmts/elf/elf-x86-x32.c
    modules/objfmts/elf/elf-x86-x86.c
    modules/objfmts/elf/elf.c
    modules/objfmts/macho/macho-objfmt.c
    modules/objfmts/rdf/rdf-objfmt.c
    modules/objfmts/xdf/xdf-objfmt.c
    modules/parsers/gas/gas-parse-intel.c
    modules/parsers/gas/gas-parse.c
    modules/parsers/gas/gas-parser.c
    modules/parsers/nasm/nasm-parse.c
    modules/parsers/nasm/nasm-parser.c
    modules/preprocs/cpp/cpp-preproc.c
    modules/preprocs/gas/gas-eval.c
    modules/preprocs/gas/gas-preproc.c
    modules/preprocs/nasm/nasm-eval.c
    modules/preprocs/nasm/nasm-pp.c
    modules/preprocs/nasm/nasm-preproc.c
    modules/preprocs/nasm/nasmlib.c
    modules/preprocs/raw/raw-preproc.c
    modules/x86cpu.c
    modules/x86regtmod.c
)

END()
