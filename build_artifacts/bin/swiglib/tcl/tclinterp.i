/* -----------------------------------------------------------------------------
 * tclinterp.i
 *
 * Tcl_Interp *interp
 *
 * Passes the current Tcl_Interp value directly to a C function.
 * This can be used to work with existing wrapper functions or
 * if you just need the interp value for some reason.  When used,
 * the 'interp' parameter becomes hidden in the Tcl interface--that
 * is, you don't specify it explicitly. SWIG fills in its value
 * automatically.
 * ----------------------------------------------------------------------------- */

%typemap(in,numinputs=0) Tcl_Interp *interp {
  $1 = interp;
}

