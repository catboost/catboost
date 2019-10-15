/* arch_lc3b arch_x86 listfmt_nasm parser_gas parser_nasm preproc_nasm preproc_raw preproc_cpp preproc_gas dbgfmt_cv8 dbgfmt_dwarf2 dbgfmt_null dbgfmt_stabs objfmt_dbg objfmt_bin objfmt_elf objfmt_elf32 objfmt_elf64 objfmt_elfx32 objfmt_coff objfmt_win32 objfmt_win64 objfmt_macho objfmt_macho32 objfmt_macho64 objfmt_rdf objfmt_xdf 
   rev 1
 */

#include <libyasm.h>
#include <libyasm/module.h>

extern yasm_arch_module yasm_lc3b_LTX_arch;
extern yasm_arch_module yasm_x86_LTX_arch;
extern yasm_listfmt_module yasm_nasm_LTX_listfmt;
extern yasm_parser_module yasm_gas_LTX_parser;
extern yasm_parser_module yasm_nasm_LTX_parser;
extern yasm_preproc_module yasm_nasm_LTX_preproc;
extern yasm_preproc_module yasm_raw_LTX_preproc;
extern yasm_preproc_module yasm_cpp_LTX_preproc;
extern yasm_preproc_module yasm_gas_LTX_preproc;
extern yasm_dbgfmt_module yasm_cv8_LTX_dbgfmt;
extern yasm_dbgfmt_module yasm_dwarf2_LTX_dbgfmt;
extern yasm_dbgfmt_module yasm_null_LTX_dbgfmt;
extern yasm_dbgfmt_module yasm_stabs_LTX_dbgfmt;
extern yasm_objfmt_module yasm_dbg_LTX_objfmt;
extern yasm_objfmt_module yasm_bin_LTX_objfmt;
extern yasm_objfmt_module yasm_elf_LTX_objfmt;
extern yasm_objfmt_module yasm_elf32_LTX_objfmt;
extern yasm_objfmt_module yasm_elf64_LTX_objfmt;
extern yasm_objfmt_module yasm_elfx32_LTX_objfmt;
extern yasm_objfmt_module yasm_coff_LTX_objfmt;
extern yasm_objfmt_module yasm_win32_LTX_objfmt;
extern yasm_objfmt_module yasm_win64_LTX_objfmt;
extern yasm_objfmt_module yasm_macho_LTX_objfmt;
extern yasm_objfmt_module yasm_macho32_LTX_objfmt;
extern yasm_objfmt_module yasm_macho64_LTX_objfmt;
extern yasm_objfmt_module yasm_rdf_LTX_objfmt;
extern yasm_objfmt_module yasm_xdf_LTX_objfmt;
void
yasm_init_plugin(void)
{
    yasm_register_module(YASM_MODULE_ARCH, "lc3b", &yasm_lc3b_LTX_arch);
    yasm_register_module(YASM_MODULE_ARCH, "x86", &yasm_x86_LTX_arch);
    yasm_register_module(YASM_MODULE_LISTFMT, "nasm", &yasm_nasm_LTX_listfmt);
    yasm_register_module(YASM_MODULE_PARSER, "gas", &yasm_gas_LTX_parser);
    yasm_register_module(YASM_MODULE_PARSER, "nasm", &yasm_nasm_LTX_parser);
    yasm_register_module(YASM_MODULE_PREPROC, "nasm", &yasm_nasm_LTX_preproc);
    yasm_register_module(YASM_MODULE_PREPROC, "raw", &yasm_raw_LTX_preproc);
    yasm_register_module(YASM_MODULE_PREPROC, "cpp", &yasm_cpp_LTX_preproc);
    yasm_register_module(YASM_MODULE_PREPROC, "gas", &yasm_gas_LTX_preproc);
    yasm_register_module(YASM_MODULE_DBGFMT, "cv8", &yasm_cv8_LTX_dbgfmt);
    yasm_register_module(YASM_MODULE_DBGFMT, "dwarf2", &yasm_dwarf2_LTX_dbgfmt);
    yasm_register_module(YASM_MODULE_DBGFMT, "null", &yasm_null_LTX_dbgfmt);
    yasm_register_module(YASM_MODULE_DBGFMT, "stabs", &yasm_stabs_LTX_dbgfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "dbg", &yasm_dbg_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "bin", &yasm_bin_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "elf", &yasm_elf_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "elf32", &yasm_elf32_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "elf64", &yasm_elf64_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "elfx32", &yasm_elfx32_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "coff", &yasm_coff_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "win32", &yasm_win32_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "win64", &yasm_win64_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "macho", &yasm_macho_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "macho32", &yasm_macho32_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "macho64", &yasm_macho64_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "rdf", &yasm_rdf_LTX_objfmt);
    yasm_register_module(YASM_MODULE_OBJFMT, "xdf", &yasm_xdf_LTX_objfmt);
}

void
yasm_plugin_set_replace(const char* replace_params[], int size) {
    yasm_dwarf2_LTX_dbgfmt.replace_map = replace_params;
    yasm_dwarf2_LTX_dbgfmt.replace_map_size = size;
    yasm_elf_LTX_objfmt.replace_map = replace_params;
    yasm_elf_LTX_objfmt.replace_map_size = size;
    yasm_elf32_LTX_objfmt.replace_map = replace_params;
    yasm_elf32_LTX_objfmt.replace_map_size = size;
    yasm_elf64_LTX_objfmt.replace_map = replace_params;
    yasm_elf64_LTX_objfmt.replace_map_size = size;
    yasm_elfx32_LTX_objfmt.replace_map = replace_params;
    yasm_elfx32_LTX_objfmt.replace_map_size = size;

    yasm_macho_LTX_objfmt.replace_map = replace_params;
    yasm_macho_LTX_objfmt.replace_map_size = size;
    yasm_macho32_LTX_objfmt.replace_map = replace_params;
    yasm_macho32_LTX_objfmt.replace_map_size = size;
    yasm_macho64_LTX_objfmt.replace_map = replace_params;
    yasm_macho64_LTX_objfmt.replace_map_size = size;
}
