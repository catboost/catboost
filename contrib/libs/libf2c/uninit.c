#include <stdio.h>
#include <string.h>
#include "arith.h"

#define TYSHORT 2
#define TYLONG 3
#define TYREAL 4
#define TYDREAL 5
#define TYCOMPLEX 6
#define TYDCOMPLEX 7
#define TYINT1 11
#define TYQUAD 14
#ifndef Long
#define Long long
#endif

#ifdef __mips
#define RNAN	0xffc00000
#define DNAN0	0xfff80000
#define DNAN1	0
#endif

#ifdef _PA_RISC1_1
#define RNAN	0xffc00000
#define DNAN0	0xfff80000
#define DNAN1	0
#endif

#ifndef RNAN
#define RNAN	0xff800001
#ifdef IEEE_MC68k
#define DNAN0	0xfff00000
#define DNAN1	1
#else
#define DNAN0	1
#define DNAN1	0xfff00000
#endif
#endif /*RNAN*/

#ifdef KR_headers
#define Void /*void*/
#define FA7UL (unsigned Long) 0xfa7a7a7aL
#else
#define Void void
#define FA7UL 0xfa7a7a7aUL
#endif

#ifdef __cplusplus
extern "C" {
#endif

static void ieee0(Void);

static unsigned Long rnan = RNAN,
	dnan0 = DNAN0,
	dnan1 = DNAN1;

double _0 = 0.;

 void
#ifdef KR_headers
_uninit_f2c(x, type, len) void *x; int type; long len;
#else
_uninit_f2c(void *x, int type, long len)
#endif
{
	static int first = 1;

	unsigned Long *lx, *lxe;

	if (first) {
		first = 0;
		ieee0();
		}
	if (len == 1)
	 switch(type) {
	  case TYINT1:
		*(char*)x = 'Z';
		return;
	  case TYSHORT:
		*(short*)x = 0xfa7a;
		break;
	  case TYLONG:
		*(unsigned Long*)x = FA7UL;
		return;
	  case TYQUAD:
	  case TYCOMPLEX:
	  case TYDCOMPLEX:
		break;
	  case TYREAL:
		*(unsigned Long*)x = rnan;
		return;
	  case TYDREAL:
		lx = (unsigned Long*)x;
		lx[0] = dnan0;
		lx[1] = dnan1;
		return;
	  default:
		printf("Surprise type %d in _uninit_f2c\n", type);
	  }
	switch(type) {
	  case TYINT1:
		memset(x, 'Z', len);
		break;
	  case TYSHORT:
		*(short*)x = 0xfa7a;
		break;
	  case TYQUAD:
		len *= 2;
		/* no break */
	  case TYLONG:
		lx = (unsigned Long*)x;
		lxe = lx + len;
		while(lx < lxe)
			*lx++ = FA7UL;
		break;
	  case TYCOMPLEX:
		len *= 2;
		/* no break */
	  case TYREAL:
		lx = (unsigned Long*)x;
		lxe = lx + len;
		while(lx < lxe)
			*lx++ = rnan;
		break;
	  case TYDCOMPLEX:
		len *= 2;
		/* no break */
	  case TYDREAL:
		lx = (unsigned Long*)x;
		for(lxe = lx + 2*len; lx < lxe; lx += 2) {
			lx[0] = dnan0;
			lx[1] = dnan1;
			}
	  }
	}
#ifdef __cplusplus
}
#endif

#ifndef MSpc
#ifdef MSDOS
#define MSpc
#else
#ifdef _WIN32
#define MSpc
#endif
#endif
#endif

#ifdef MSpc
#define IEEE0_done
#include "float.h"
#include "signal.h"

 static void
ieee0(Void)
{
#ifndef __alpha
#ifndef EM_DENORMAL
#define EM_DENORMAL _EM_DENORMAL
#endif
#ifndef EM_UNDERFLOW
#define EM_UNDERFLOW _EM_UNDERFLOW
#endif
#ifndef EM_INEXACT
#define EM_INEXACT _EM_INEXACT
#endif
#ifndef MCW_EM
#define MCW_EM _MCW_EM
#endif
	_control87(EM_DENORMAL | EM_UNDERFLOW | EM_INEXACT, MCW_EM);
#endif
	/* With MS VC++, compiling and linking with -Zi will permit */
	/* clicking to invoke the MS C++ debugger, which will show */
	/* the point of error -- provided SIGFPE is SIG_DFL. */
	signal(SIGFPE, SIG_DFL);
	}
#endif /* MSpc */

#ifdef __mips	/* must link with -lfpe */
#define IEEE0_done
/* code from Eric Grosse */
#include <stdlib.h>
#include <stdio.h>
#error #include "/usr/include/sigfpe.h"	/* full pathname for lcc -N */
#error #include "/usr/include/sys/fpu.h"

 static void
#ifdef KR_headers
ieeeuserhand(exception, val) unsigned exception[5]; int val[2];
#else
ieeeuserhand(unsigned exception[5], int val[2])
#endif
{
	fflush(stdout);
	fprintf(stderr,"ieee0() aborting because of ");
	if(exception[0]==_OVERFL) fprintf(stderr,"overflow\n");
	else if(exception[0]==_UNDERFL) fprintf(stderr,"underflow\n");
	else if(exception[0]==_DIVZERO) fprintf(stderr,"divide by 0\n");
	else if(exception[0]==_INVALID) fprintf(stderr,"invalid operation\n");
	else fprintf(stderr,"\tunknown reason\n");
	fflush(stderr);
	abort();
}

 static void
#ifdef KR_headers
ieeeuserhand2(j) unsigned int **j;
#else
ieeeuserhand2(unsigned int **j)
#endif
{
	fprintf(stderr,"ieee0() aborting because of confusion\n");
	abort();
}

 static void
ieee0(Void)
{
	int i;
	for(i=1; i<=4; i++){
		sigfpe_[i].count = 1000;
		sigfpe_[i].trace = 1;
		sigfpe_[i].repls = _USER_DETERMINED;
		}
	sigfpe_[1].repls = _ZERO;	/* underflow */
	handle_sigfpes( _ON,
		_EN_UNDERFL|_EN_OVERFL|_EN_DIVZERO|_EN_INVALID,
		ieeeuserhand,_ABORT_ON_ERROR,ieeeuserhand2);
	}
#endif /* mips */

#if 0
#ifdef __linux__
#define IEEE0_done
#include "fpu_control.h"

#ifdef __alpha__
#ifndef USE_setfpucw
#define __setfpucw(x) __fpu_control = (x)
#endif
#endif

#ifndef _FPU_SETCW
#undef  Can_use__setfpucw
#define Can_use__setfpucw
#endif

 static void
ieee0(Void)
{
#if (defined(__mc68000__) || defined(__mc68020__) || defined(mc68020) || defined (__mc68k__))
/* Reported 20010705 by Alan Bain <alanb@chiark.greenend.org.uk> */
/* Note that IEEE 754 IOP (illegal operation) */
/* = Signaling NAN (SNAN) + operation error (OPERR). */
#ifdef Can_use__setfpucw
	__setfpucw(_FPU_IEEE + _FPU_DOUBLE + _FPU_MASK_OPERR + _FPU_MASK_DZ + _FPU_MASK_SNAN+_FPU_MASK_OVFL);
#else
	__fpu_control = _FPU_IEEE + _FPU_DOUBLE + _FPU_MASK_OPERR + _FPU_MASK_DZ + _FPU_MASK_SNAN+_FPU_MASK_OVFL;
	_FPU_SETCW(__fpu_control);
#endif

#elif (defined(__powerpc__)||defined(_ARCH_PPC)||defined(_ARCH_PWR)) /* !__mc68k__ */
/* Reported 20011109 by Alan Bain <alanb@chiark.greenend.org.uk> */

#ifdef Can_use__setfpucw

/* The following is NOT a mistake -- the author of the fpu_control.h
for the PPC has erroneously defined IEEE mode to turn on exceptions
other than Inexact! Start from default then and turn on only the ones
which we want*/

	__setfpucw(_FPU_DEFAULT +  _FPU_MASK_IM+_FPU_MASK_OM+_FPU_MASK_UM);

#else /* PPC && !Can_use__setfpucw */

	__fpu_control = _FPU_DEFAULT +_FPU_MASK_OM+_FPU_MASK_IM+_FPU_MASK_UM;
	_FPU_SETCW(__fpu_control);

#endif /*Can_use__setfpucw*/

#else /* !(mc68000||powerpc) */

#ifdef _FPU_IEEE
#ifndef _FPU_EXTENDED /* e.g., ARM processor under Linux */
#define _FPU_EXTENDED 0
#endif
#ifndef _FPU_DOUBLE
#define _FPU_DOUBLE 0
#endif
#ifdef Can_use__setfpucw /* pre-1997 (?) Linux */
	__setfpucw(_FPU_IEEE - _FPU_MASK_IM - _FPU_MASK_ZM - _FPU_MASK_OM);
#else
#ifdef UNINIT_F2C_PRECISION_53 /* 20051004 */
	/* unmask invalid, etc., and change rounding precision to double */
	__fpu_control = _FPU_IEEE - _FPU_EXTENDED + _FPU_DOUBLE - _FPU_MASK_IM - _FPU_MASK_ZM - _FPU_MASK_OM;
	_FPU_SETCW(__fpu_control);
#else
	/* unmask invalid, etc., and keep current rounding precision */
	fpu_control_t cw;
	_FPU_GETCW(cw);
	cw &= ~(_FPU_MASK_IM | _FPU_MASK_ZM | _FPU_MASK_OM);
	_FPU_SETCW(cw);
#endif
#endif

#else /* !_FPU_IEEE */

	fprintf(stderr, "\n%s\n%s\n%s\n%s\n",
		"WARNING:  _uninit_f2c in libf2c does not know how",
		"to enable trapping on this system, so f2c's -trapuv",
		"option will not detect uninitialized variables unless",
		"you can enable trapping manually.");
	fflush(stderr);

#endif /* _FPU_IEEE */
#endif /* __mc68k__ */
	}
#endif /* __linux__ */
#endif

#ifdef __alpha
#ifndef IEEE0_done
#define IEEE0_done
#include <machine/fpu.h>
 static void
ieee0(Void)
{
	ieee_set_fp_control(IEEE_TRAP_ENABLE_INV);
	}
#endif /*IEEE0_done*/
#endif /*__alpha*/

#ifdef __hpux
#define IEEE0_done
#define _INCLUDE_HPUX_SOURCE
#include <math.h>

#ifndef FP_X_INV
#include <fenv.h>
#define fpsetmask fesettrapenable
#define FP_X_INV FE_INVALID
#endif

 static void
ieee0(Void)
{
	fpsetmask(FP_X_INV);
	}
#endif /*__hpux*/

#ifdef _AIX
#define IEEE0_done
#include <fptrap.h>

 static void
ieee0(Void)
{
	fp_enable(TRP_INVALID);
	fp_trap(FP_TRAP_SYNC);
	}
#endif /*_AIX*/

#ifdef __sun
#define IEEE0_done
#include <ieeefp.h>

 static void
ieee0(Void)
{
	fpsetmask(FP_X_INV);
	}
#endif /*__sparc*/

#ifndef IEEE0_done
 static void
ieee0(Void) {}
#endif
