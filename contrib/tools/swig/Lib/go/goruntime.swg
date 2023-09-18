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

%insert(cgo_comment_typedefs) %{
#include <stddef.h>
#include <stdint.h>
%}

#if SWIGGO_INTGO_SIZE == 32
%insert(runtime) %{
typedef int intgo;
typedef unsigned int uintgo;
%}
%insert(cgo_comment_typedefs) %{
typedef int intgo;
typedef unsigned int uintgo;
%}
#elif SWIGGO_INTGO_SIZE == 64
%insert(runtime) %{
typedef long long intgo;
typedef unsigned long long uintgo;
%}
%insert(cgo_comment_typedefs) %{
typedef long long intgo;
typedef unsigned long long uintgo;
%}
#else
%insert(runtime) %{
typedef ptrdiff_t intgo;
typedef size_t uintgo;
%}
%insert(cgo_comment_typedefs) %{
typedef ptrdiff_t intgo;
typedef size_t uintgo;
%}
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

%insert(cgo_comment_typedefs) %{

typedef struct { char *p; intgo n; } _gostring_;
typedef struct { void* array; intgo len; intgo cap; } _goslice_;

%}

#ifdef SWIGGO_GCCGO

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

#endif

#ifndef SWIGGO_GCCGO

%go_import("unsafe", _ "runtime/cgo")

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

/* Convert a Go interface value into a C++ pointer.  */

%insert(go_header) %{
func getSwigcptr(v interface { Swigcptr() uintptr }) uintptr {
	if v == nil {
		return 0
	}
	return v.Swigcptr()
}
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
