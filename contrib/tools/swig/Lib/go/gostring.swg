/* ------------------------------------------------------------
 * gostring.swg
 *
 * Support for returning strings from C to Go.
 * ------------------------------------------------------------ */

// C/C++ code to convert a memory buffer into a Go string allocated in
// C/C++ memory.
%fragment("AllocateString", "runtime") %{
static _gostring_ Swig_AllocateString(const char *p, size_t l) {
  _gostring_ ret;
  ret.p = (char*)malloc(l);
  memcpy(ret.p, p, l);
  ret.n = l;
  return ret;
}
%}

// Go code to convert a string allocated in C++ memory to one
// allocated in Go memory.
%fragment("CopyString", "go_runtime") %{
type swig_gostring struct { p unsafe.Pointer; n int }
func swigCopyString(s string) string {
  p := *(*swig_gostring)(unsafe.Pointer(&s))
  r := string((*[0x7fffffff]byte)(p.p)[:p.n])
  Swig_free(uintptr(p.p))
  return r
}
%}
