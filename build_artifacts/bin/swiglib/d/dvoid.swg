/* -----------------------------------------------------------------------------
 * dvoid.swg
 *
 * Typemaps for handling void function return types and empty parameter lists.
 * ----------------------------------------------------------------------------- */

%typemap(ctype) void "void"
%typemap(imtype) void "void"
%typemap(dtype, cprimitive="1") void "void"

%typemap(out, null="") void ""
%typemap(ddirectorin) void "$winput"
%typemap(ddirectorout) void "$dcall"
%typemap(directorin) void ""

%typemap(dout, excode=SWIGEXCODE) void {
  $imcall;$excode
}
