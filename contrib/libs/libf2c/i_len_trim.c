#include "f2c.h"
#ifdef __cplusplus
extern "C" {
#endif

#ifdef KR_headers
integer i_len_trim(s, n) char *s; ftnlen n;
#else
integer i_len_trim(char *s, ftnlen n)
#endif
{
  int i;

  for(i=n-1;i>=0;i--)
    if(s[i] != ' ')
      return i + 1;

  return(0);
}
#ifdef __cplusplus
}
#endif
