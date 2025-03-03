%{
#if defined(__linux__)
#include <endian.h>
#if BYTE_ORDER == LITTLE_ENDIAN
#define SWIG_RUBY_ENDIAN "LE"
#elif BYTE_ORDER == BIG_ENDIAN
#define SWIG_RUBY_ENDIAN "BE"
#endif
#else
#define SWIG_RUBY_ENDIAN "LE"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_RUBY_ENCODING_H
#include "ruby/encoding.h"
#endif

/**
 *  The internal encoding of std::wstring is defined based on
 *  the size of wchar_t. If it is not appropriate for your library,
 *  SWIG_RUBY_WSTRING_ENCODING must be given when compiling.
 */
#ifndef SWIG_RUBY_WSTRING_ENCODING

#if WCHAR_MAX == 0x7fff || WCHAR_MAX == 0xffff
#define SWIG_RUBY_WSTRING_ENCODING "UTF-16" SWIG_RUBY_ENDIAN
#elif WCHAR_MAX == 0x7fffffff || WCHAR_MAX == 0xffffffff
#define SWIG_RUBY_WSTRING_ENCODING "UTF-32" SWIG_RUBY_ENDIAN
#else
#error unsupported wchar_t size. SWIG_RUBY_WSTRING_ENCODING must be given.
#endif

#endif

/**
 *  If Encoding.default_internal is nil, this encoding will be used
 *  when converting from std::wstring to String object in Ruby.
 */
#ifndef SWIG_RUBY_INTERNAL_ENCODING
#define SWIG_RUBY_INTERNAL_ENCODING "UTF-8"
#endif

static rb_encoding *swig_ruby_wstring_encoding;
static rb_encoding *swig_ruby_internal_encoding;

#ifdef __cplusplus
}
#endif
%}

%fragment("SWIG_ruby_wstring_encoding_init", "init") {
  swig_ruby_wstring_encoding  = rb_enc_find( SWIG_RUBY_WSTRING_ENCODING );
  swig_ruby_internal_encoding = rb_enc_find( SWIG_RUBY_INTERNAL_ENCODING );
}

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) std::basic_string<wchar_t>;

%include <rubywstrings.swg>
%include <typemaps/std_wstring.swg>

