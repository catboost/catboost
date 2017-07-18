#ifndef SHGETC_H
#define SHGETC_H

#include <stdio.h>
#include <wchar.h>

// Custom stream abstraction, replaces the Musl FILE type for
// the purpose of scanning integers and floating points.
// |rstart| is the start of the input string.
// |rend| is the first wchar_t after it.
// |rpos| is the current reading position.
// |extra_eof| is a counter of positions past EOF. Needed because the
// scanning routines more or less assume an infinite input string, with
// EOF being returned when shgetc() is being called past the real end
// of the input stream.
struct fake_file_t {
    const wchar_t *rstart, *rpos, *rend;
    int extra_eof;
};

// Initialize fake_file_t structure from a wide-char string.
void shinit_wcstring(struct fake_file_t *, const wchar_t *wcs);

// Get next character from string. This convers the wide character to
// an 8-bit value, which will be '@' in case of overflow. Returns EOF (-1)
// in case of end-of-string.
int shgetc(struct fake_file_t *);

// Back-track one character, must not be called more times than shgetc()
void shunget(struct fake_file_t *);

// This will be called with a value of 0 for |lim| to force rewinding
// to the start of the string. In Musl, this is also used in different
// parts of the library to impose a local limit on the number of characters
// that can be retrieved through shgetc(), but this is not necessary here.
void shlim(struct fake_file_t *, off_t lim);

// Return the number of input characters that were read so far.
off_t shcnt(struct fake_file_t *);

#endif  // SHGETC_H
