/* Character handling in C locale.

   These functions work like the corresponding functions in <ctype.h>,
   except that they have the C (POSIX) locale hardwired, whereas the
   <ctype.h> functions' behaviour depends on the current locale set via
   setlocale.

   Copyright (C) 2000-2003, 2006, 2008-2020 Free Software Foundation, Inc.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifndef C_CTYPE_H
#define C_CTYPE_H

#include <stdbool.h>

#ifndef _GL_INLINE_HEADER_BEGIN
 #error "Please include config.h first."
#endif
_GL_INLINE_HEADER_BEGIN
#ifndef C_CTYPE_INLINE
# define C_CTYPE_INLINE _GL_INLINE
#endif

#ifdef __cplusplus
extern "C" {
#endif


/* The functions defined in this file assume the "C" locale and a character
   set without diacritics (ASCII-US or EBCDIC-US or something like that).
   Even if the "C" locale on a particular system is an extension of the ASCII
   character set (like on BeOS, where it is UTF-8, or on AmigaOS, where it
   is ISO-8859-1), the functions in this file recognize only the ASCII
   characters.  */


#if (' ' == 32) && ('!' == 33) && ('"' == 34) && ('#' == 35) \
    && ('%' == 37) && ('&' == 38) && ('\'' == 39) && ('(' == 40) \
    && (')' == 41) && ('*' == 42) && ('+' == 43) && (',' == 44) \
    && ('-' == 45) && ('.' == 46) && ('/' == 47) && ('0' == 48) \
    && ('1' == 49) && ('2' == 50) && ('3' == 51) && ('4' == 52) \
    && ('5' == 53) && ('6' == 54) && ('7' == 55) && ('8' == 56) \
    && ('9' == 57) && (':' == 58) && (';' == 59) && ('<' == 60) \
    && ('=' == 61) && ('>' == 62) && ('?' == 63) && ('A' == 65) \
    && ('B' == 66) && ('C' == 67) && ('D' == 68) && ('E' == 69) \
    && ('F' == 70) && ('G' == 71) && ('H' == 72) && ('I' == 73) \
    && ('J' == 74) && ('K' == 75) && ('L' == 76) && ('M' == 77) \
    && ('N' == 78) && ('O' == 79) && ('P' == 80) && ('Q' == 81) \
    && ('R' == 82) && ('S' == 83) && ('T' == 84) && ('U' == 85) \
    && ('V' == 86) && ('W' == 87) && ('X' == 88) && ('Y' == 89) \
    && ('Z' == 90) && ('[' == 91) && ('\\' == 92) && (']' == 93) \
    && ('^' == 94) && ('_' == 95) && ('a' == 97) && ('b' == 98) \
    && ('c' == 99) && ('d' == 100) && ('e' == 101) && ('f' == 102) \
    && ('g' == 103) && ('h' == 104) && ('i' == 105) && ('j' == 106) \
    && ('k' == 107) && ('l' == 108) && ('m' == 109) && ('n' == 110) \
    && ('o' == 111) && ('p' == 112) && ('q' == 113) && ('r' == 114) \
    && ('s' == 115) && ('t' == 116) && ('u' == 117) && ('v' == 118) \
    && ('w' == 119) && ('x' == 120) && ('y' == 121) && ('z' == 122) \
    && ('{' == 123) && ('|' == 124) && ('}' == 125) && ('~' == 126)
/* The character set is ASCII or one of its variants or extensions, not EBCDIC.
   Testing the value of '\n' and '\r' is not relevant.  */
# define C_CTYPE_ASCII 1
#elif ! (' ' == '\x40' && '0' == '\xf0'                     \
         && 'A' == '\xc1' && 'J' == '\xd1' && 'S' == '\xe2' \
         && 'a' == '\x81' && 'j' == '\x91' && 's' == '\xa2')
# error "Only ASCII and EBCDIC are supported"
#endif

#if 'A' < 0
# error "EBCDIC and char is signed -- not supported"
#endif

/* Cases for control characters.  */

#define _C_CTYPE_CNTRL \
   case '\a': case '\b': case '\f': case '\n': \
   case '\r': case '\t': case '\v': \
   _C_CTYPE_OTHER_CNTRL

/* ASCII control characters other than those with \-letter escapes.  */

#if C_CTYPE_ASCII
# define _C_CTYPE_OTHER_CNTRL \
    case '\x00': case '\x01': case '\x02': case '\x03': \
    case '\x04': case '\x05': case '\x06': case '\x0e': \
    case '\x0f': case '\x10': case '\x11': case '\x12': \
    case '\x13': case '\x14': case '\x15': case '\x16': \
    case '\x17': case '\x18': case '\x19': case '\x1a': \
    case '\x1b': case '\x1c': case '\x1d': case '\x1e': \
    case '\x1f': case '\x7f'
#else
   /* Use EBCDIC code page 1047's assignments for ASCII control chars;
      assume all EBCDIC code pages agree about these assignments.  */
# define _C_CTYPE_OTHER_CNTRL \
    case '\x00': case '\x01': case '\x02': case '\x03': \
    case '\x07': case '\x0e': case '\x0f': case '\x10': \
    case '\x11': case '\x12': case '\x13': case '\x18': \
    case '\x19': case '\x1c': case '\x1d': case '\x1e': \
    case '\x1f': case '\x26': case '\x27': case '\x2d': \
    case '\x2e': case '\x32': case '\x37': case '\x3c': \
    case '\x3d': case '\x3f'
#endif

/* Cases for lowercase hex letters, and lowercase letters, all offset by N.  */

#define _C_CTYPE_LOWER_A_THRU_F_N(N) \
   case 'a' + (N): case 'b' + (N): case 'c' + (N): case 'd' + (N): \
   case 'e' + (N): case 'f' + (N)
#define _C_CTYPE_LOWER_N(N) \
   _C_CTYPE_LOWER_A_THRU_F_N(N): \
   case 'g' + (N): case 'h' + (N): case 'i' + (N): case 'j' + (N): \
   case 'k' + (N): case 'l' + (N): case 'm' + (N): case 'n' + (N): \
   case 'o' + (N): case 'p' + (N): case 'q' + (N): case 'r' + (N): \
   case 's' + (N): case 't' + (N): case 'u' + (N): case 'v' + (N): \
   case 'w' + (N): case 'x' + (N): case 'y' + (N): case 'z' + (N)

/* Cases for hex letters, digits, lower, punct, and upper.  */

#define _C_CTYPE_A_THRU_F \
   _C_CTYPE_LOWER_A_THRU_F_N (0): \
   _C_CTYPE_LOWER_A_THRU_F_N ('A' - 'a')
#define _C_CTYPE_DIGIT                     \
   case '0': case '1': case '2': case '3': \
   case '4': case '5': case '6': case '7': \
   case '8': case '9'
#define _C_CTYPE_LOWER _C_CTYPE_LOWER_N (0)
#define _C_CTYPE_PUNCT \
   case '!': case '"': case '#': case '$':  \
   case '%': case '&': case '\'': case '(': \
   case ')': case '*': case '+': case ',':  \
   case '-': case '.': case '/': case ':':  \
   case ';': case '<': case '=': case '>':  \
   case '?': case '@': case '[': case '\\': \
   case ']': case '^': case '_': case '`':  \
   case '{': case '|': case '}': case '~'
#define _C_CTYPE_UPPER _C_CTYPE_LOWER_N ('A' - 'a')


/* Function definitions.  */

/* Unlike the functions in <ctype.h>, which require an argument in the range
   of the 'unsigned char' type, the functions here operate on values that are
   in the 'unsigned char' range or in the 'char' range.  In other words,
   when you have a 'char' value, you need to cast it before using it as
   argument to a <ctype.h> function:

         const char *s = ...;
         if (isalpha ((unsigned char) *s)) ...

   but you don't need to cast it for the functions defined in this file:

         const char *s = ...;
         if (c_isalpha (*s)) ...
 */

C_CTYPE_INLINE bool
c_isalnum (int c)
{
  switch (c)
    {
    _C_CTYPE_DIGIT:
    _C_CTYPE_LOWER:
    _C_CTYPE_UPPER:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isalpha (int c)
{
  switch (c)
    {
    _C_CTYPE_LOWER:
    _C_CTYPE_UPPER:
      return true;
    default:
      return false;
    }
}

/* The function isascii is not locale dependent.
   Its use in EBCDIC is questionable. */
C_CTYPE_INLINE bool
c_isascii (int c)
{
  switch (c)
    {
    case ' ':
    _C_CTYPE_CNTRL:
    _C_CTYPE_DIGIT:
    _C_CTYPE_LOWER:
    _C_CTYPE_PUNCT:
    _C_CTYPE_UPPER:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isblank (int c)
{
  return c == ' ' || c == '\t';
}

C_CTYPE_INLINE bool
c_iscntrl (int c)
{
  switch (c)
    {
    _C_CTYPE_CNTRL:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isdigit (int c)
{
  switch (c)
    {
    _C_CTYPE_DIGIT:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isgraph (int c)
{
  switch (c)
    {
    _C_CTYPE_DIGIT:
    _C_CTYPE_LOWER:
    _C_CTYPE_PUNCT:
    _C_CTYPE_UPPER:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_islower (int c)
{
  switch (c)
    {
    _C_CTYPE_LOWER:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isprint (int c)
{
  switch (c)
    {
    case ' ':
    _C_CTYPE_DIGIT:
    _C_CTYPE_LOWER:
    _C_CTYPE_PUNCT:
    _C_CTYPE_UPPER:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_ispunct (int c)
{
  switch (c)
    {
    _C_CTYPE_PUNCT:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isspace (int c)
{
  switch (c)
    {
    case ' ': case '\t': case '\n': case '\v': case '\f': case '\r':
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isupper (int c)
{
  switch (c)
    {
    _C_CTYPE_UPPER:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE bool
c_isxdigit (int c)
{
  switch (c)
    {
    _C_CTYPE_DIGIT:
    _C_CTYPE_A_THRU_F:
      return true;
    default:
      return false;
    }
}

C_CTYPE_INLINE int
c_tolower (int c)
{
  switch (c)
    {
    _C_CTYPE_UPPER:
      return c - 'A' + 'a';
    default:
      return c;
    }
}

C_CTYPE_INLINE int
c_toupper (int c)
{
  switch (c)
    {
    _C_CTYPE_LOWER:
      return c - 'a' + 'A';
    default:
      return c;
    }
}

#ifdef __cplusplus
}
#endif

_GL_INLINE_HEADER_END

#endif /* C_CTYPE_H */
