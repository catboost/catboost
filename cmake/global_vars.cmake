
# This file was gererated by the build system used internally in the Yandex monorepo.
# Only simple modifications are allowed (adding source-files to targets, adding simple properties
# like target_include_directories). These modifications will be ported to original
# ya.make files by maintainers. Any complex modifications which can't be ported back to the
# original buildsystem will not be accepted.


if(UNIX AND NOT APPLE)
  set(FAT_OBJECT_SUFFIX .o)
  set(FAT_OBJECT_PREFIX lib)
  set(YASM_FLAGS -f elf64 -D UNIX -D _x86_64_ -D_YASM_ -g dwarf2)
  set(FBS_CPP_FLAGS --no-warnings --cpp --keep-prefix --gen-mutable --schema -b --yandex-maps-iter --gen-object-api --filename-suffix .fbs)
  set(RAGEL_FLAGS -L -I ${CMAKE_SOURCE_DIR}/)
endif()

if(APPLE)
  set(FAT_OBJECT_SUFFIX .o)
  set(FAT_OBJECT_PREFIX lib)
  set(YASM_FLAGS -f macho64 -D DARWIN -D UNIX -D _x86_64_ -D_YASM_)
  set(FBS_CPP_FLAGS --no-warnings --cpp --keep-prefix --gen-mutable --schema -b --yandex-maps-iter --gen-object-api --filename-suffix .fbs)
  set(RAGEL_FLAGS -L -I ${CMAKE_SOURCE_DIR}/)
endif()

