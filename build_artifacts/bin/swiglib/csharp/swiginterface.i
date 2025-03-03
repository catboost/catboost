/* -----------------------------------------------------------------------------
 * swiginterface.i
 *
 * SWIG interface feature and typemaps implementation providing:
 *   %interface
 *   %interface_impl
 *   %interface_custom
 * ----------------------------------------------------------------------------- */

%define INTERFACE_TYPEMAPS(CTYPE...)
%typemap(cstype) CTYPE "$&csinterfacename"
%typemap(cstype) CTYPE *, CTYPE [], CTYPE & "$csinterfacename"
%typemap(cstype) CTYPE *const& "$*csinterfacename"
%typemap(csin) CTYPE, CTYPE & "$csinput.GetInterfaceCPtr()"
%typemap(csin) CTYPE *, CTYPE *const&, CTYPE [] "$csinput == null ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : $csinput.GetInterfaceCPtr()"
%typemap(csout, excode=SWIGEXCODE) CTYPE {
    $&csclassname ret = new $&csclassname($imcall, true);$excode
    return ($&csinterfacename)ret;
  }
%typemap(csout, excode=SWIGEXCODE) CTYPE & {
    $csclassname ret = new $csclassname($imcall, $owner);$excode
    return ($csinterfacename)ret;
  }
%typemap(csout, excode=SWIGEXCODE) CTYPE *, CTYPE [] {
    global::System.IntPtr cPtr = $imcall;
    $csclassname ret = (cPtr == global::System.IntPtr.Zero) ? null : new $csclassname(cPtr, $owner);$excode
    return ($csinterfacename)ret;
  }
%typemap(csout, excode=SWIGEXCODE) CTYPE *const& {
    global::System.IntPtr cPtr = $imcall;
    $*csclassname ret = (cPtr == global::System.IntPtr.Zero) ? null : new $*csclassname(cPtr, $owner);$excode
    return ($*csinterfacename)ret;
  }
%typemap(csdirectorin) CTYPE "($&csinterfacename)new $&csclassname($iminput, true)"
%typemap(csdirectorin) CTYPE & "($csinterfacename)new $csclassname($iminput, false)"
%typemap(csdirectorin) CTYPE *, CTYPE [] "($iminput == global::System.IntPtr.Zero) ? null : ($csinterfacename)new $csclassname($iminput, false)"
%typemap(csdirectorin) CTYPE *const& "($iminput == global::System.IntPtr.Zero) ? null : ($*csinterfacename)new $*csclassname($iminput, false)"
%typemap(csdirectorout) CTYPE, CTYPE *, CTYPE *const&, CTYPE [], CTYPE & "$cscall.GetInterfaceCPtr()"
%typemap(csinterfacecode, declaration="  [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]\n  global::System.Runtime.InteropServices.HandleRef GetInterfaceCPtr();\n", cptrmethod="$interfacename_GetInterfaceCPtr") CTYPE %{
  [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
  global::System.Runtime.InteropServices.HandleRef $interfacename.GetInterfaceCPtr() {
    return new global::System.Runtime.InteropServices.HandleRef(this, $imclassname.$csclazzname$interfacename_GetInterfaceCPtr(swigCPtr.Handle));
  }
%}
%enddef

%define %interface(CTYPE...)
%feature("interface", name="%sSwigInterface") CTYPE;
INTERFACE_TYPEMAPS(CTYPE)
%enddef

%define %interface_impl(CTYPE...)
%rename("%sSwigImpl") CTYPE;
%feature("interface", name="%(rstrip:[SwigImpl])s") CTYPE;
INTERFACE_TYPEMAPS(CTYPE)
%enddef

%define %interface_custom(PROXY, INTERFACE, CTYPE...)
%rename(PROXY) CTYPE;
%feature("interface", name=INTERFACE) CTYPE;
INTERFACE_TYPEMAPS(CTYPE)
%enddef

