/* -----------------------------------------------------------------------------
 * tclresult.i
 * ----------------------------------------------------------------------------- */

/*
int Tcl_Result

      Makes the integer return code of a function the return value 
      of a SWIG generated wrapper function.  For example :

            int foo() {
                  ... do stuff ...
                  return TCL_OK;
            }      

      could be wrapped as follows :

            %include typemaps.i
            %apply int Tcl_Result { int foo };
            int foo();
*/

// If return code is a Tcl_Result, simply pass it on

%typemap(out) int Tcl_Result {
  return $1;
}
