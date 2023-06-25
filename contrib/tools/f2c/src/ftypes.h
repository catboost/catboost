
/* variable types (stored in the   vtype  field of   expptr)
 * numeric assumptions:
 *	int < reals < complexes
 *	TYDREAL-TYREAL = TYDCOMPLEX-TYCOMPLEX
 */

#undef TYQUAD0
#ifdef NO_TYQUAD
#undef TYQUAD
#define TYQUAD_inc 0
#undef NO_LONG_LONG
#define NO_LONG_LONG
#else
#define TYQUAD 5
#define TYQUAD_inc 1
#ifdef NO_LONG_LONG
#define TYQUAD0
#else
#ifndef Llong
typedef long long Llong;
#endif
#ifndef ULlong
typedef unsigned long long ULlong;
#endif
#endif /*NO_LONG_LONG*/
#endif /*NO_TYQUAD*/

#ifdef _WIN64
#define USE_LONGLONG
#endif

#ifdef USE_LONGLONG
typedef unsigned long long Addr;
#define Addrfmt "%llx"
#define Atol atoll
#else
typedef unsigned long Addr;
#define Addrfmt "%lx"
#define Atol atol
#endif

#define TYUNKNOWN 0
#define TYADDR 1
#define TYINT1 2
#define TYSHORT 3
#define TYLONG 4
/* #define TYQUAD 5 */
#define TYREAL (5+TYQUAD_inc)
#define TYDREAL (6+TYQUAD_inc)
#define TYCOMPLEX (7+TYQUAD_inc)
#define TYDCOMPLEX (8+TYQUAD_inc)
#define TYLOGICAL1 (9+TYQUAD_inc)
#define TYLOGICAL2 (10+TYQUAD_inc)
#define TYLOGICAL (11+TYQUAD_inc)
#define TYCHAR (12+TYQUAD_inc)
#define TYSUBR (13+TYQUAD_inc)
#define TYERROR (14+TYQUAD_inc)
#define TYCILIST (15+TYQUAD_inc)
#define TYICILIST (16+TYQUAD_inc)
#define TYOLIST (17+TYQUAD_inc)
#define TYCLLIST (18+TYQUAD_inc)
#define TYALIST (19+TYQUAD_inc)
#define TYINLIST (20+TYQUAD_inc)
#define TYVOID (21+TYQUAD_inc)
#define TYLABEL (22+TYQUAD_inc)
#define TYFTNLEN (23+TYQUAD_inc)
/* TYVOID is not in any tables. */

/* NTYPES, NTYPES0 -- Total number of types, used to allocate tables indexed by
   type.  Such tables can include the size (in bytes) of objects of a given
   type, or labels for returning objects of different types from procedures
   (see array   rtvlabels)   */

#define NTYPES TYVOID
#define NTYPES0 TYCILIST
#define TYBLANK TYSUBR		/* Huh? */

