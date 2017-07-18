#include "blaswrap.h"
#include "f2c.h"
#include <sys/times.h>
#include <sys/types.h>
#include <time.h>

#ifndef CLK_TCK
#define CLK_TCK 60
#endif

doublereal dsecnd_()
{
  struct tms rusage;

  times(&rusage);
  return (doublereal)(rusage.tms_utime) / CLK_TCK;

} /* dsecnd_ */
