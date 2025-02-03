/*
 * Copyright 2018-2022 Yury Gribov
 *
 * The MIT License (MIT)
 *
 * Use of this source code is governed by MIT license that can be
 * found in the LICENSE.txt file.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // For RTLD_DEFAULT
#endif

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

// Sanity check for ARM to avoid puzzling runtime crashes
#ifdef __arm__
# if defined __thumb__ && ! defined __THUMB_INTERWORK__
#   error "ARM trampolines need -mthumb-interwork to work in Thumb mode"
# endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK(cond, fmt, ...) do { \
    if(!(cond)) { \
      fprintf(stderr, "implib-gen: $load_name: " fmt "\n", ##__VA_ARGS__); \
      assert(0 && "Assertion in generated code"); \
      abort(); \
    } \
  } while(0)

#define HAS_DLOPEN_CALLBACK $has_dlopen_callback
#define HAS_DLSYM_CALLBACK $has_dlsym_callback
#define NO_DLOPEN $no_dlopen
#define LAZY_LOAD $lazy_load

static void *lib_handle;
static int do_dlclose;
static int is_lib_loading;

#if ! NO_DLOPEN
static void *load_library() {
  if(lib_handle)
    return lib_handle;

  is_lib_loading = 1;

#if HAS_DLOPEN_CALLBACK
  extern void *$dlopen_callback(const char *lib_name);
  lib_handle = $dlopen_callback("$load_name");
  CHECK(lib_handle, "failed to load library '$load_name' via callback '$dlopen_callback'");
#else
  lib_handle = dlopen("$load_name", RTLD_LAZY | RTLD_GLOBAL);
  CHECK(lib_handle, "failed to load library '$load_name' via dlopen: %s", dlerror());
#endif

  do_dlclose = 1;
  is_lib_loading = 0;

  return lib_handle;
}

static void __attribute__((destructor)) unload_lib() {
  if(do_dlclose && lib_handle)
    dlclose(lib_handle);
}
#endif

#if ! NO_DLOPEN && ! LAZY_LOAD
static void __attribute__((constructor)) load_lib() {
  load_library();
}
#endif

// TODO: convert to single 0-separated string
static const char *const sym_names[] = {
  $sym_names
  0
};

#define SYM_COUNT (sizeof(sym_names)/sizeof(sym_names[0]) - 1)

extern void *_${lib_suffix}_tramp_table[];

// Can be sped up by manually parsing library symtab...
void _${lib_suffix}_tramp_resolve(int i) {
  assert((unsigned)i < SYM_COUNT);

  CHECK(!is_lib_loading, "library function '%s' called during library load", sym_names[i]);

  void *h = 0;
#if NO_DLOPEN
  // Library with implementations must have already been loaded.
  if (lib_handle) {
    // User has specified loaded library
    h = lib_handle;
  } else {
    // User hasn't provided us the loaded library so search the global namespace.
#   ifndef IMPLIB_EXPORT_SHIMS
    // If shim symbols are hidden we should search
    // for first available definition of symbol in library list
    h = RTLD_DEFAULT;
#   else
    // Otherwise look for next available definition
    h = RTLD_NEXT;
#   endif
  }
#else
  h = load_library();
  CHECK(h, "failed to resolve symbol '%s', library failed to load", sym_names[i]);
#endif

#if HAS_DLSYM_CALLBACK
  extern void *$dlsym_callback(void *handle, const char *sym_name);
  _${lib_suffix}_tramp_table[i] = $dlsym_callback(h, sym_names[i]);
  CHECK(_${lib_suffix}_tramp_table[i], "failed to resolve symbol '%s' via callback $dlsym_callback", sym_names[i]);
#else
  // Dlsym is thread-safe so don't need to protect it.
  _${lib_suffix}_tramp_table[i] = dlsym(h, sym_names[i]);
  CHECK(_${lib_suffix}_tramp_table[i], "failed to resolve symbol '%s' via dlsym: %s", sym_names[i], dlerror());
#endif
}

// Helper for user to resolve all symbols
void _${lib_suffix}_tramp_resolve_all(void) {
  size_t i;
  for(i = 0; i < SYM_COUNT; ++i)
    _${lib_suffix}_tramp_resolve(i);
}

// Allows user to specify manually loaded implementation library.
void _${lib_suffix}_tramp_set_handle(void *handle) {
  lib_handle = handle;
  do_dlclose = 0;
}

// Resets all resolved symbols. This is needed in case
// client code wants to reload interposed library multiple times.
void _${lib_suffix}_tramp_reset(void) {
  memset(_${lib_suffix}_tramp_table, 0, SYM_COUNT * sizeof(_${lib_suffix}_tramp_table[0]));
  lib_handle = 0;
  do_dlclose = 0;
}

#ifdef __cplusplus
}  // extern "C"
#endif
