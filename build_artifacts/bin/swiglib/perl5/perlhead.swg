#ifdef __cplusplus
/* Needed on some windows machines---since MS plays funny games with the header files under C++ */
#include <math.h>
#include <stdlib.h>
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

/* Add in functionality missing in older versions of Perl. Much of this is based on Devel-PPPort on cpan. */

/* Add PERL_REVISION, PERL_VERSION, PERL_SUBVERSION if missing */
#ifndef PERL_REVISION
#  if !defined(__PATCHLEVEL_H_INCLUDED__) && !(defined(PATCHLEVEL) && defined(SUBVERSION))
#    define PERL_PATCHLEVEL_H_IMPLICIT
#    include <patchlevel.h>
#  endif
#  if !(defined(PERL_VERSION) || (defined(SUBVERSION) && defined(PATCHLEVEL)))
#    include <could_not_find_Perl_patchlevel.h>
#  endif
#  ifndef PERL_REVISION
#    define PERL_REVISION       (5)
#    define PERL_VERSION        PATCHLEVEL
#    define PERL_SUBVERSION     SUBVERSION
#  endif
#endif

#if defined(WIN32) && defined(PERL_OBJECT) && !defined(PerlIO_exportFILE)
#define PerlIO_exportFILE(fh,fl) (FILE*)(fh)
#endif

#ifndef SvIOK_UV
# define SvIOK_UV(sv)       (SvIOK(sv) && (SvUVX(sv) == SvIVX(sv)))
#endif

#ifndef SvUOK
# define SvUOK(sv)           SvIOK_UV(sv)
#endif

#if ((PERL_VERSION < 4) || ((PERL_VERSION == 4) && (PERL_SUBVERSION <= 5)))
#  define PL_sv_undef               sv_undef
#  define PL_na	                    na
#  define PL_errgv                  errgv
#  define PL_sv_no                  sv_no
#  define PL_sv_yes                 sv_yes
#  define PL_markstack_ptr          markstack_ptr
#endif

#ifndef IVSIZE
#  ifdef LONGSIZE
#    define IVSIZE LONGSIZE
#  else
#    define IVSIZE 4 /* A bold guess, but the best we can make. */
#  endif
#endif

#ifndef INT2PTR
#  if (IVSIZE == PTRSIZE) && (UVSIZE == PTRSIZE)
#    define PTRV                  UV
#    define INT2PTR(any,d)        (any)(d)
#  else
#    if PTRSIZE == LONGSIZE
#      define PTRV                unsigned long
#    else
#      define PTRV                unsigned
#    endif
#    define INT2PTR(any,d)        (any)(PTRV)(d)
#  endif

#  define NUM2PTR(any,d)  (any)(PTRV)(d)
#  define PTR2IV(p)       INT2PTR(IV,p)
#  define PTR2UV(p)       INT2PTR(UV,p)
#  define PTR2NV(p)       NUM2PTR(NV,p)

#  if PTRSIZE == LONGSIZE
#    define PTR2ul(p)     (unsigned long)(p)
#  else
#    define PTR2ul(p)     INT2PTR(unsigned long,p)
#  endif
#endif /* !INT2PTR */

#ifndef SvPV_nolen
# define SvPV_nolen(x) SvPV(x,PL_na)
#endif

#ifndef get_sv
#  define get_sv perl_get_sv
#endif

#ifndef ERRSV
#  define ERRSV get_sv("@",FALSE)
#endif

#ifndef pTHX_
#define pTHX_
#endif   

#include <string.h>
#ifdef __cplusplus
}
#endif
