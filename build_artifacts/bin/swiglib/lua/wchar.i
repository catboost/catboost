/* -----------------------------------------------------------------------------
 * wchar.i
 *
 * Typemaps for the wchar_t type
 * These are mapped to a Lua string and are passed around by value.
 * ----------------------------------------------------------------------------- */

// note: only support for pointer right now, not fixed length strings
// TODO: determine how long a const wchar_t* is so we can write wstr2str() 
// & do the output typemap

%{
#include <stdlib.h>
	
wchar_t* str2wstr(const char *str, int len)
{
  wchar_t* p;
  if (str==0 || len<1)  return 0;
  p=(wchar_t *)malloc((len+1)*sizeof(wchar_t));
  if (p==0)	return 0;
  if (mbstowcs(p, str, len)==(size_t)-1)
  {
    free(p);
    return 0;
  }
  p[len]=0;
  return p;
}
%}

%typemap(in, checkfn="SWIG_lua_isnilstring", fragment="SWIG_lua_isnilstring") wchar_t *
%{
$1 = str2wstr(lua_tostring( L, $input ),lua_rawlen( L, $input ));
if ($1==0) {SWIG_Lua_pushferrstring(L,"Error in converting to wchar (arg %d)",$input);goto fail;}
%}

%typemap(freearg) wchar_t *
%{
free($1);
%}

%typemap(typecheck) wchar_t * = char *;
