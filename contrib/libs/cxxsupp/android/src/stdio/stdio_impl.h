// The Musl sources for snprintf(), sscanf() assume they can use a specially
// crafted FILE object to represent the output/input buffers. However, this
// doesn't work when using FILE handle from Bionic.
//
// This header is used to 'cheat' by redefining FILE and a few other macro
// redefinitions for functions used in the sources in this directory.

#ifndef STDIO_IMPL_H
#define STDIO_IMPL_H

#define __HIDDEN__  __attribute__((__visibility__("hidden")))

// A structure that wraps either a real FILE* handle, or an input/output
// buffer.
typedef struct {
  FILE* file;
  unsigned char* buffer;
  size_t buffer_size;
  size_t buffer_pos;
} FakeFILE;

// Initialize FakeFILE wrapper |file| to use a FILE* handle |f|
void fake_file_init_file(FakeFILE* file, FILE* f) __HIDDEN__;

// Initialize FakeFILE wrapper |file| to use a |buffer| of |buffer_size| chars.
void fake_file_init_buffer(FakeFILE* file, char* buffer, size_t buffer_size)
    __HIDDEN__;

// Initialize FakeFILE wrapper |file| to use a wchar_t |buffer| of
// |buffer_size| wide-chars.
void fake_file_init_wbuffer(FakeFILE* file, wchar_t* buffer, size_t buffer_size)
    __HIDDEN__;

// Replacement for out() in vfprintf.c
void fake_file_out(FakeFILE* file, const char* s, size_t l) __HIDDEN__;

// Replacement for out() in fvwprintf.c
void fake_file_outw(FakeFILE* file, const wchar_t* s, size_t l) __HIDDEN__;

// Fake replacement for stdio functions of similar names.
int fake_feof(FakeFILE* file) __HIDDEN__;
int fake_ferror(FakeFILE* file) __HIDDEN__;
int fake_fprintf(FakeFILE* file, const char* fmt, ...) __HIDDEN__;
void fake_fputc(char ch, FakeFILE* file) __HIDDEN__;
void fake_fputwc(wchar_t wc, FakeFILE* file) __HIDDEN__;

#ifndef _STDIO_IMPL_NO_REDIRECT_MACROS

// Macro redirection - ugly but necessary to minimize changes to the sources.
#define FILE FakeFILE

#undef feof
#define feof fake_feof

#undef ferror
#define ferror fake_ferror
#define fprintf fake_fprintf
#define fputc fake_fputc
#define fputwc fake_fputwc

#endif  /* _STDIO_IMPL_NO_REDIRECT_MACROS */

#endif  /* STDIO_IMPL_H */
