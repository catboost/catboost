/* -----------------------------------------------------------------------------
 * tclsh.i
 *
 * SWIG File for building new tclsh program
 * ----------------------------------------------------------------------------- */

#ifdef AUTODOC
%subsection "tclsh.i"
%text %{
This module provides the Tcl_AppInit() function needed to build a 
new version of the tclsh executable.   This file should not be used
when using dynamic loading.   To make an interface file work with
both static and dynamic loading, put something like this in your
interface file :

     #ifdef STATIC
     %include <tclsh.i>
     #endif
%}
#endif

%{

/* A TCL_AppInit() function that lets you build a new copy
 * of tclsh.
 *
 * The macro SWIG_init contains the name of the initialization
 * function in the wrapper file.
 */

#ifndef SWIG_RcFileName
char *SWIG_RcFileName = "~/.myapprc";
#endif


#ifdef MAC_TCL
extern int		MacintoshInit _ANSI_ARGS_((void));
#endif

int Tcl_AppInit(Tcl_Interp *interp){

  if (Tcl_Init(interp) == TCL_ERROR) 
    return TCL_ERROR;

  /* Now initialize our functions */

  if (SWIG_init(interp) == TCL_ERROR)
    return TCL_ERROR;
#if TCL_MAJOR_VERSION > 7 || TCL_MAJOR_VERSION == 7 && TCL_MINOR_VERSION >= 5
   Tcl_SetVar(interp, (char *) "tcl_rcFileName",SWIG_RcFileName,TCL_GLOBAL_ONLY);
#else
   tcl_RcFileName = SWIG_RcFileName;
#endif
#ifdef SWIG_RcRsrcName
  Tcl_SetVar(interp, (char *) "tcl_rcRsrcName",SWIG_RcRsrcName,TCL_GLOBAL);
#endif
  
  return TCL_OK;
}

#if TCL_MAJOR_VERSION > 7 || TCL_MAJOR_VERSION == 7 && TCL_MINOR_VERSION >= 4
int main(int argc, char **argv) {
#ifdef MAC_TCL
    char *newArgv[2];
    
    if (MacintoshInit()  != TCL_OK) {
	Tcl_Exit(1);
    }

    argc = 1;
    newArgv[0] = "tclsh";
    newArgv[1] = NULL;
    argv = newArgv;
#endif

  Tcl_Main(argc, argv, Tcl_AppInit);
  return(0);

}
#else
extern int main();
#endif

%}

