/* ------------------------------------------------------------
 * goruntime.swg
 *
 * Go runtime code for the various generated files.
 * ------------------------------------------------------------ */

%inline %{
static void Swig_free(void* p) {
  free(p);
}

static void* Swig_malloc(int c) {
  return malloc(c);
}
%}

%insert(runtime) %{
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

%}

#if SWIGGO_CGO
%insert(cgo_comment_typedefs) %{
#include <stdint.h>
%}
#endif

#if SWIGGO_INTGO_SIZE == 32
%insert(runtime) %{
typedef int intgo;
typedef unsigned int uintgo;
%}
#if SWIGGO_CGO
%insert(cgo_comment_typedefs) %{
typedef int intgo;
typedef unsigned int uintgo;
%}
#endif
#elif SWIGGO_INTGO_SIZE == 64
%insert(runtime) %{
typedef long long intgo;
typedef unsigned long long uintgo;
%}
#if SWIGGO_CGO
%insert(cgo_comment_typedefs) %{
typedef long long intgo;
typedef unsigned long long uintgo;
%}
#endif
#else
%insert(runtime) %{
typedef ptrdiff_t intgo;
typedef size_t uintgo;
%}
#if SWIGGO_CGO
%insert(cgo_comment_typedefs) %{
typedef ptrdiff_t intgo;
typedef size_t uintgo;
%}
#endif
#endif

#ifndef SWIGGO_GCCGO
// Set the host compiler struct attribute that will be
// used to match gc's struct layout. For example, on 386 Windows,
// gcc wants to 8-align int64s, but gc does not.
// Use __gcc_struct__ to work around http://gcc.gnu.org/PR52991 on x86,
// and https://golang.org/issue/5603.
// See: https://github.com/golang/go/blob/fcbf04f9b93b4cd8addd05c2ed784118eb50a46c/src/cmd/cgo/out.go#L663
%insert(runtime) %{
# if !defined(__clang__) && (defined(__i386__) || defined(__x86_64__))
#   define SWIGSTRUCTPACKED __attribute__((__packed__, __gcc_struct__))
# else
#   define SWIGSTRUCTPACKED __attribute__((__packed__))
# endif
%}
#else
# define SWIGSTRUCTPACKED
#endif

%insert(runtime) %{

typedef struct { char *p; intgo n; } _gostring_;
typedef struct { void* array; intgo len; intgo cap; } _goslice_;

%}

#ifdef SWIGGO_CGO

%insert(cgo_comment_typedefs) %{

typedef struct { char *p; intgo n; } _gostring_;
typedef struct { void* array; intgo len; intgo cap; } _goslice_;

%}

#endif

#ifndef SWIGGO_GCCGO
/* Boilerplate for C/C++ code when using 6g/8g.  This code is compiled
   with gcc.  */
%insert(runtime) %{

#define swiggo_size_assert_eq(x, y, name) typedef char name[(x-y)*(x-y)*-2+1];
#define swiggo_size_assert(t, n) swiggo_size_assert_eq(sizeof(t), n, swiggo_sizeof_##t##_is_not_##n)

swiggo_size_assert(char, 1)
swiggo_size_assert(short, 2)
swiggo_size_assert(int, 4)
typedef long long swiggo_long_long;
swiggo_size_assert(swiggo_long_long, 8)
swiggo_size_assert(float, 4)
swiggo_size_assert(double, 8)

#ifdef __cplusplus
extern "C" {
#endif
extern void crosscall2(void (*fn)(void *, int), void *, int);
extern char* _cgo_topofstack(void) __attribute__ ((weak));
extern void _cgo_allocate(void *, int);
extern void _cgo_panic(void *, int);
#ifdef __cplusplus
}
#endif

static char *_swig_topofstack() {
  if (_cgo_topofstack) {
    return _cgo_topofstack();
  } else {
    return 0;
  }
}

static void _swig_gopanic(const char *p) {
  struct {
    const char *p;
  } SWIGSTRUCTPACKED a;
  a.p = p;
  crosscall2(_cgo_panic, &a, (int) sizeof a);
}

%}

#if !SWIGGO_CGO

/* This is here for backward compatibility, but it will not work
   with Go 1.5 or later.  Do not use it in new code.  */
%insert(runtime) %{

static void *_swig_goallocate(size_t len) {
  struct {
    size_t len;
    void *ret;
  } SWIGSTRUCTPACKED a;
  a.len = len;
  crosscall2(_cgo_allocate, &a, (int) sizeof a);
  return a.ret;
}

%}

#endif

#if !SWIGGO_CGO

/* Boilerplate for C code when using 6g/8g.  This code is compiled
   with 6c/8c.  */
%insert(gc_header) %{
#include "runtime.h"
#include "cgocall.h"

#pragma dataflag 16
static void *cgocall = runtime·cgocall;
#pragma dataflag 16
void *·_cgo_runtime_cgocall = &cgocall;

%}

#endif

#else

/* Boilerplate for C/C++ code when using gccgo.  */
%insert(runtime) %{
#define SWIGGO_GCCGO

#ifdef __cplusplus
extern "C" {
#endif
extern void *_cgo_allocate(size_t);
extern void _cgo_panic(const char *);
#ifdef __cplusplus
}
#endif

#define _swig_goallocate _cgo_allocate
#define _swig_gopanic _cgo_panic
%}

#if !SWIGGO_CGO

%insert(runtime) %{

/* Implementations of SwigCgocall and friends for different versions
   of gccgo.  The Go code will call these functions using C names with
   a prefix of the module name.  The implementations here call the
   routine in libgo.  The routines to call vary depending on the gccgo
   version.  We assume that the version of gcc used to compile this
   file is the same as the version of gccgo.  */

#ifdef __cplusplus
extern "C" {
#endif

#define SWIG_GCC_VERSION \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

#if SWIG_GCC_VERSION < 40700
#define SwigDoCgocall()
#define SwigDoCgocallDone()
#define SwigDoCgocallBack()
#define SwigDoCgocallBackDone()
#elif SWIG_GCC_VERSION == 40700
void SwigDoCgocall(void) __asm__("libgo_syscall.syscall.Entersyscall");
void SwigDoCgocallDone(void) __asm__("libgo_syscall.syscall.Exitsyscall");
void SwigDoCgocallBack(void) __asm__("libgo_syscall.syscall.Exitsyscall");
void SwigDoCgocallBackDone(void) __asm__("libgo_syscall.syscall.Entersyscall");
#else
void SwigDoCgocall(void) __asm__("syscall.Cgocall");
void SwigDoCgocallDone(void) __asm__("syscall.CgocallDone");
void SwigDoCgocallBack(void) __asm__("syscall.CgocallBack");
void SwigDoCgocallBackDone(void) __asm__("syscall.CgocallBackDone");
#endif

#define SWIGSTRINGIFY2(s) #s
#define SWIGSTRINGIFY(s) SWIGSTRINGIFY2(s)

void SwigCgocall()
  __asm__(SWIGSTRINGIFY(SWIGGO_PREFIX) ".SwigCgocall");
void SwigCgocall() {
  SwigDoCgocall();
}

void SwigCgocallDone()
  __asm__(SWIGSTRINGIFY(SWIGGO_PREFIX) ".SwigCgocallDone");
void SwigCgocallDone() {
  SwigDoCgocallDone();
}

void SwigCgocallBack()
  __asm__(SWIGSTRINGIFY(SWIGGO_PREFIX) ".SwigCgocallBack");
void SwigCgocallBack() {
  SwigDoCgocallBack();
}

void SwigCgocallBackDone()
  __asm__(SWIGSTRINGIFY(SWIGGO_PREFIX) ".SwigCgocallBackDone");
void SwigCgocallBackDone() {
  SwigDoCgocallBackDone();
}

#undef SWIGSTRINGIFY
#undef SWIGSTRINGIFY2

#ifdef __cplusplus
}
#endif

%}

#endif

#endif

#if !SWIGGO_CGO

%insert(runtime) %{

/* This is here for backward compatibility, but it will not work
   with Go 1.5 or later.  Do not use it in new code.  */
static _gostring_ _swig_makegostring(const char *p, size_t l) {
  _gostring_ ret;
  ret.p = (char*)_swig_goallocate(l + 1);
  memcpy(ret.p, p, l);
  ret.n = l;
  return ret;
}

%}

#endif

%insert(runtime) %{

#define SWIG_contract_assert(expr, msg) \
  if (!(expr)) { _swig_gopanic(msg); } else
%}

#ifndef SWIGGO_GCCGO

%go_import("unsafe", _ "runtime/cgo")

#if !SWIGGO_CGO
%insert(go_header) %{
var _cgo_runtime_cgocall func(unsafe.Pointer, uintptr)
%}
#endif

#else

%go_import("syscall", "unsafe")

%insert(go_header) %{

type _ syscall.Sockaddr

%}

#endif

%insert(go_header) %{

type _ unsafe.Pointer

%}

/* Swig_always_false is used to conditionally assign parameters to
   Swig_escape_val so that the compiler thinks that they escape.  We
   only assign them if Swig_always_false is true, which it never is.
   We export the variable so that the compiler doesn't realize that it
   is never set. */
%insert(go_header) %{
var Swig_escape_always_false bool
var Swig_escape_val interface{}
%}

/* Function pointers are translated by the code in go.cxx into
   _swig_fnptr.  Member pointers are translated to _swig_memberptr.  */

%insert(go_header) %{
type _swig_fnptr *byte
type _swig_memberptr *byte
%}

/* For directors we need C++ to track a Go pointer.  Since we can't
   pass a Go pointer into C++, we use a map to track the pointers on
   the Go side.  */

%go_import("sync")

%insert(go_header) %{
type _ sync.Mutex
%}

%insert(go_director) %{

var swigDirectorTrack struct {
	sync.Mutex
	m map[int]interface{}
	c int
}

func swigDirectorAdd(v interface{}) int {
	swigDirectorTrack.Lock()
	defer swigDirectorTrack.Unlock()
	if swigDirectorTrack.m == nil {
		swigDirectorTrack.m = make(map[int]interface{})
	}
	swigDirectorTrack.c++
	ret := swigDirectorTrack.c
	swigDirectorTrack.m[ret] = v
	return ret
}

func swigDirectorLookup(c int) interface{} {
	swigDirectorTrack.Lock()
	defer swigDirectorTrack.Unlock()
	ret := swigDirectorTrack.m[c]
	if ret == nil {
		panic("C++ director pointer not found (possible	use-after-free)")
	}
	return ret
}

func swigDirectorDelete(c int) {
	swigDirectorTrack.Lock()
	defer swigDirectorTrack.Unlock()
	if swigDirectorTrack.m[c] == nil {
		if c > swigDirectorTrack.c {
			panic("C++ director pointer invalid (possible memory corruption")
		} else {
			panic("C++ director pointer not found (possible use-after-free)")
		}
	}
	delete(swigDirectorTrack.m, c)
}

%}
