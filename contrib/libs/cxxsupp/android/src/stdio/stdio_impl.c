#include <stdio.h>
#include <wchar.h>
#include <stdlib.h>
#include <string.h>

#define _STDIO_IMPL_NO_REDIRECT_MACROS
#include "stdio_impl.h"

void fake_file_init_file(FakeFILE* file, FILE* f) {
  memset(file, 0, sizeof(*file));
  file->file = f;
}

void fake_file_init_buffer(FakeFILE* file, char* buffer, size_t buffer_size) {
  memset(file, 0, sizeof(*file));
  file->buffer = (void*)buffer;
  file->buffer_pos = 0;
  file->buffer_size = buffer_size;
}

void fake_file_init_wbuffer(FakeFILE* file,
                            wchar_t* buffer,
                            size_t buffer_size) {
  fake_file_init_buffer(file, (char*)buffer, buffer_size * sizeof(wchar_t));
}

void fake_file_out(FakeFILE* file, const char* text, size_t length) {
  if (length == 0) {
    // Happens pretty often, so bail immediately.
    return;
  }
  if (file->file != NULL) {
    fwrite(text, 1, length, file->file);
  } else {
    // Write into a bounded buffer.
    size_t avail = file->buffer_size - file->buffer_pos;
    if (length > avail)
      length = avail;
    memcpy((char*)(file->buffer + file->buffer_pos),
           (const char*)text,
           length);
    file->buffer_pos += length;
  }
}

void fake_file_outw(FakeFILE* file, const wchar_t* text, size_t length) {
  if (length == 0)
    return;
  if (file->file != NULL) {
    // Write into a file the UTF-8 encoded version.
    // Original source calls fputwc() in a loop, which is slightly inefficient
    // for large strings.
    // TODO(digit): Support locale-specific encoding?
    size_t mb_len = wcstombs(NULL, text, length);
    char* mb_buffer = malloc(mb_len);
    wcstombs(mb_buffer, text, length);
    fwrite(mb_buffer, 1, mb_len, file->file);
    free(mb_buffer);
  } else {
    // Write into a bounded buffer. This assumes the buffer really
    // holds wchar_t items.
    size_t avail = (file->buffer_size - file->buffer_pos) / sizeof(wchar_t);
    if (length > avail)
      length = avail;
    memcpy((char*)(file->buffer + file->buffer_pos),
           (const char*)text,
           length * sizeof(wchar_t));
    file->buffer_pos += length * sizeof(wchar_t);
  }
}

int fake_feof(FakeFILE* file) {
  if (file->file != NULL)
    return feof(file->file);
  else
    return (file->buffer_pos >= file->buffer_size);
}

int fake_ferror(FakeFILE* file) {
  if (file->file != NULL)
    return ferror(file->file);

  return 0;
}

int fake_fprintf(FakeFILE* file, const char* format, ...) {
  va_list args;
  va_start(args, format);
  if (file->file)
    return vfprintf(file->file, format, args);
  else {
    // TODO(digit): Make this faster.
    // First, generate formatted byte output.
    int mb_len = vsnprintf(NULL, 0, format, args);
    char* mb_buffer = malloc(mb_len + 1);
    vsnprintf(mb_buffer, mb_len + 1, format, args);
    // Then convert to wchar_t buffer.
    size_t wide_len = mbstowcs(NULL, mb_buffer, mb_len);
    wchar_t* wide_buffer = malloc((wide_len + 1) * sizeof(wchar_t));
    mbstowcs(wide_buffer, mb_buffer, mb_len);
    // Add to buffer.
    fake_file_outw(file, wide_buffer, wide_len);
    // finished
    free(wide_buffer);
    free(mb_buffer);

    return wide_len;
  }
  va_end(args);
}

void fake_fputc(char ch, FakeFILE* file) {
  if (file->file)
    fputc(ch, file->file);
  else {
    if (file->buffer_pos < file->buffer_size)
      file->buffer[file->buffer_pos++] = ch;
  }
}

void fake_fputwc(wchar_t wc, FakeFILE* file) {
  if (file->file)
    fputwc(wc, file->file);
  else {
    if (file->buffer_pos + sizeof(wchar_t) - 1U < file->buffer_size) {
      *(wchar_t*)(&file->buffer[file->buffer_pos]) = wc;
      file->buffer_pos += sizeof(wchar_t);
    }
  }
}
