#!/usr/bin/env python

import base64
import hashlib
import io
import os
import struct
import subprocess
import sys
from collections import namedtuple


"""
ELF-64 Object File Format: https://uclibc.org/docs/elf-64-gen.pdf
ELF-32: https://uclibc.org/docs/elf.pdf
"""


MANGLED_HASH_SIZE = 15

# len(base64(sha1(name)[:MANGLED_HASH_SIZE]) + '\x00')
MANGLED_NAME_SIZE = 21


ArObject = namedtuple('ArObject', ['header', 'data'])

ElfSection = namedtuple('ElfSection', ['header_offset', 'name', 'data_offset', 'size', 'link', 'entry_size'])


def find(it, pred):
    return next(iter(filter(pred, it)), None)


def mangle_name(name_bytes):
    sha1 = hashlib.sha1()
    sha1.update(name_bytes)
    dgst = sha1.digest()
    return base64.b64encode(dgst[:MANGLED_HASH_SIZE])


def unpack(format, buffer, offset=0):
    return struct.unpack(format, buffer[offset : offset + struct.calcsize(format)])


def unpack_section_header(buffer, offset, elf64):
    # read sh_name, sh_offset, sh_size, sh_link, sh_entsize from section headers (Elf64_Shdr/Elf32_Shdr):
    #
    # typedef struct
    # {
    #     Elf64_Word  sh_name;        /* Section name */
    #     Elf64_Word  sh_type;        /* Section type */
    #     Elf64_Xword sh_flags;       /* Section attributes */
    #     Elf64_Addr  sh_addr;        /* Virtual address in memory */
    #     Elf64_Off   sh_offset;      /* Offset in file */
    #     Elf64_Xword sh_size;        /* Size of section */
    #     Elf64_Word  sh_link;        /* Link to other section */
    #     Elf64_Word  sh_info;        /* Miscellaneous information */
    #     Elf64_Xword sh_addralign;   /* Address alignment boundary */
    #     Elf64_Xword sh_entsize;     /* Size of entries, if section has table */
    # } Elf64_Shdr;
    #
    # typedef struct {
    #     Elf32_Word  sh_name;
    #     Elf32_Word  sh_type;
    #     Elf32_Word  sh_flags;
    #     Elf32_Addr  sh_addr;
    #     Elf32_Off   sh_offset;
    #     Elf32_Word  sh_size;
    #     Elf32_Word  sh_link;
    #     Elf32_Word  sh_info;
    #     Elf32_Word  sh_addralign;
    #     Elf32_Word  sh_entsize;
    # } Elf32_Shdr;

    section_header_format = '< L 20x Q Q L 12x Q' if elf64 else '< L 12x L L L 8x L'
    return ElfSection(offset, *unpack(section_header_format, buffer, offset))


def read_elf_sections(elf_data, elf64):
    # read e_shoff, e_shentsize, e_shnum, e_shstrndx from elf header (Elf64_Ehdr/Elf32_Ehdr):
    #
    # typedef struct
    # {
    #     unsigned char   e_ident[16];    /* ELF identification */
    #     Elf64_Half      e_type;         /* Object file type */
    #     Elf64_Half      e_machine;      /* Machine type */
    #     Elf64_Word      e_version;      /* Object file version */
    #     Elf64_Addr      e_entry;        /* Entry point address */
    #     Elf64_Off       e_phoff;        /* Program header offset */
    #     Elf64_Off       e_shoff;        /* Section header offset */
    #     Elf64_Word      e_flags;        /* Processor-specific flags */
    #     Elf64_Half      e_ehsize;       /* ELF header size */
    #     Elf64_Half      e_phentsize;    /* Size of program header entry */
    #     Elf64_Half      e_phnum;        /* Number of program header entries */
    #     Elf64_Half      e_shentsize;    /* Size of section header entry */
    #     Elf64_Half      e_shnum;        /* Number of section header entries */
    #     Elf64_Half      e_shstrndx;     /* Section name string table index */
    # } Elf64_Ehdr;
    #
    # #define EI_NIDENT   16
    #
    # typedef struct {
    #     unsigned char   e_ident[EI_NIDENT];
    #     Elf32_Half      e_type;
    #     Elf32_Half      e_machine;
    #     Elf32_Word      e_version;
    #     Elf32_Addr      e_entry;
    #     Elf32_Off       e_phoff;
    #     Elf32_Off       e_shoff;
    #     Elf32_Word      e_flags;
    #     Elf32_Half      e_ehsize;
    #     Elf32_Half      e_phentsize;
    #     Elf32_Half      e_phnum;
    #     Elf32_Half      e_shentsize;
    #     Elf32_Half      e_shnum;
    #     Elf32_Half      e_shstrndx;
    # } Elf32_Ehdr;

    section_header_offset, section_header_entry_size, section_header_entries_number,\
        section_name_string_table_index = unpack('< Q 10x 3H', elf_data, 40) if elf64 else unpack('< L 10x 3H', elf_data, 32)

    # https://refspecs.linuxfoundation.org/elf/gabi4+/ch4.sheader.html
    # If the number of sections is greater than or equal to SHN_LORESERVE (0xff00), e_shnum has the value SHN_UNDEF (0)
    # and the actual number of section header table entries is contained in the sh_size field of the section header
    # at index 0 (otherwise, the sh_size member of the initial entry contains 0).
    if section_header_entries_number == 0:
        section_header_entries_number = unpack_section_header(elf_data, section_header_offset, elf64).size

    sections = [unpack_section_header(elf_data, section_header_offset + i * section_header_entry_size, elf64)
                for i in range(section_header_entries_number)]

    # section names data
    section_names_section = sections[section_name_string_table_index]
    section_names_data = elf_data[section_names_section.data_offset : section_names_section.data_offset + section_names_section.size]

    # read section names
    for i, section in enumerate(sections):
        sections[i] = section._replace(
            name=section_names_data[section.name : section_names_data.find(b'\x00', section.name)].decode())

    return sections


def mangle_elf_typeinfo_names(elf_data, elf64, sections):
    symbol_sizes = {}

    for sect_i, section in enumerate(sections):
        if not section.name.startswith('.rodata._ZTS') or section.size <= MANGLED_NAME_SIZE:
            continue

        typeinfo_name = elf_data[section.data_offset : section.data_offset + section.size]
        mangled = mangle_name(typeinfo_name.rstrip(b'\x00')) + b'\x00'
        if len(mangled) >= len(typeinfo_name):
            continue

        # patch section data
        elf_data[section.data_offset : section.data_offset + len(mangled)] = mangled
        # patch section size (sh_size in Elf64_Shdr/Elf32_Shdr)
        if elf64:
            elf_data[section.header_offset + 32 : section.header_offset + 40] = struct.pack('< Q', len(mangled))
        else:
            elf_data[section.header_offset + 20 : section.header_offset + 24] = struct.pack('< L', len(mangled))

        symbol_sizes[section.name[len('.rodata.'):]] = len(mangled)

    return symbol_sizes


def patch_elf_symbol_sizes(elf_data, elf64, sections, symbol_sizes):
    symtab = find(sections, lambda s: s.name == '.symtab')
    if not symtab:
        return

    for sym_i in range(symtab.size / symtab.entry_size):
        symtab_entry_offset = symtab.data_offset + symtab.entry_size * sym_i
        symtab_entry = elf_data[symtab_entry_offset : symtab_entry_offset + symtab.entry_size]

        # unpack symbol name offset in symbols name section (st_name) from Elf64_Sym/Elf32_Sym:
        #
        # typedef struct
        # {
        #     Elf64_Word      st_name;    /* Symbol name */
        #     unsigned char   st_info;    /* Type and Binding attributes */
        #     unsigned char   st_other;   /* Reserved */
        #     Elf64_Half      st_shndx;   /* Section table index */
        #     Elf64_Addr      st_value;   /* Symbol value */
        #     Elf64_Xword     st_size;    /* Size of object (e.g., common) */
        # } Elf64_Sym;
        #
        # typedef struct {
        #     Elf32_Word      st_name;
        #     Elf32_Addr      st_value;
        #     Elf32_Word      st_size;
        #     unsigned char   st_info;
        #     unsigned char   st_other;
        #     Elf32_Half      st_shndx;
        # } Elf32_Sym;
        symbol_name_offset = unpack('< L', symtab_entry)[0]

        # symbol name offset from start of elf file
        global_name_offset = sections[symtab.link].data_offset + symbol_name_offset

        name = elf_data[global_name_offset : elf_data.find(b'\x00', global_name_offset)].decode()
        symbol_size = symbol_sizes.get(name)
        if symbol_size:
            # patch st_size in Elf64_Sym/Elf32_Sym
            if elf64:
                elf_data[symtab_entry_offset + 16 : symtab_entry_offset + 24] = struct.pack('< Q', symbol_size)
            else:
                elf_data[symtab_entry_offset + 8 : symtab_entry_offset + 12] = struct.pack('< L', symbol_size)


def mangle_elf(elf_data):
    elf_data = bytearray(elf_data)

    ei_mag, ei_class = unpack('4s B', elf_data)
    assert ei_mag == b'\x7fELF'
    if ei_class == 1: # ELFCLASS32
        elf64 = False
    elif ei_class == 2: # ELFCLASS64
        elf64 = True
    else:
        raise Exception('unknown ei_class: ' + str(ei_class))

    sections = read_elf_sections(elf_data, elf64)

    symbol_sizes = mangle_elf_typeinfo_names(elf_data, elf64, sections)

    if len(symbol_sizes) != 0:
        patch_elf_symbol_sizes(elf_data, elf64, sections, symbol_sizes)

    return elf_data


def read_ar_object(ar):
    # ar format: https://docs.oracle.com/cd/E36784_01/html/E36873/ar.h-3head.html
    #
    # #define  ARFMAG   "`\n"         /* header trailer string */
    #
    # struct  ar_hdr                  /* file member header */
    # {
    #     char    ar_name[16];        /* '/' terminated file member name */
    #     char    ar_date[12];        /* file member date */
    #     char    ar_uid[6]           /* file member user identification */
    #     char    ar_gid[6]           /* file member group identification */
    #     char    ar_mode[8]          /* file member mode (octal) */
    #     char    ar_size[10];        /* file member size */
    #     char    ar_fmag[2];         /* header trailer string */
    # };

    header = ar.read(60)
    if len(header) == 0:
        return None
    assert header[58:] == b'`\n'

    size = int(bytes(header[48:58]).decode().rstrip(' '))
    data = ar.read(size)
    return ArObject(header, data)


def is_elf_data(data):
    return data[:4] == b'\x7fELF'


def mangle_ar_impl(ar, out):
    ar_magic = ar.read(8)
    if ar_magic != b'!<arch>\n':
        raise Exception('bad ar magic: {}'.format(ar_magic))

    out.write(ar_magic)

    string_table = None

    while True:
        obj = read_ar_object(ar)
        if not obj:
            break

        data = mangle_elf(obj.data) if is_elf_data(obj.data) else obj.data

        out.write(obj.header)
        out.write(data)


def mangle_ar(path):
    out_path = path + '.mangled'
    with open(path, 'rb') as ar:
        try:
            with open(out_path, 'wb') as out:
                mangle_ar_impl(ar, out)
        except:
            os.unlink(out_path)
            raise

        os.rename(out_path, path)


def main():
    for arg in sys.argv[1:]:
        if not ((arg.endswith('.o') or arg.endswith('.a')) and os.path.exists(arg)):
            continue

        if arg.endswith('.o'):
            with open(arg, 'rb') as o:
                data = o.read()
                mangled = mangle_elf(data) if is_elf_data(data) else None

            if mangled:
                os.unlink(arg)
                with open(arg, 'wb') as o:
                    o.write(mangled)
        elif arg.endswith('.a'):
            mangle_ar(arg)


if __name__ == '__main__':
    main()
