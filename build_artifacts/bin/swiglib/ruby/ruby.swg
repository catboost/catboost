/* ------------------------------------------------------------
 * ruby.swg
 *
 * Ruby configuration module.
 * ------------------------------------------------------------ */

/* ------------------------------------------------------------
 * The Ruby auto rename rules 
 * ------------------------------------------------------------ */
#if defined(SWIG_RUBY_AUTORENAME)
/* Class names are CamelCase */
%rename("%(camelcase)s", %$isclass) ""; 

/* Constants created by %constant or #define are UPPER_CASE */
%rename("%(uppercase)s", %$isconstant) "";

/* SWIG only considers static class members with inline initializers
	 to be constants.  For examples of what is and isn't considered
	 a constant by SWIG see naming.i in the Ruby test suite. */
%rename("%(uppercase)s", %$ismember, %$isvariable,%$isimmutable,%$isstatic,%$hasvalue,%$hasconsttype) ""; 

/* Enums are mapped to constants but all we do is make sure the
   first letter is uppercase */
%rename("%(firstuppercase)s", %$isenumitem) "";

/* Method names should be lower_case_with_underscores */
%rename("%(undercase)s", %$isfunction, %$not %$ismemberget, %$not %$ismemberset) "";
#endif

/* ------------------------------------------------------------
 *  Inner macros 
 * ------------------------------------------------------------ */
%include <rubymacros.swg>


/* ------------------------------------------------------------
 *  The runtime part
 * ------------------------------------------------------------ */
%include <rubyruntime.swg>

/* ------------------------------------------------------------
 *  Special user directives
 * ------------------------------------------------------------ */
%include <rubyuserdir.swg>

/* ------------------------------------------------------------
 *  Typemap specializations
 * ------------------------------------------------------------ */
%include <rubytypemaps.swg>

/* ------------------------------------------------------------
 *  Overloaded operator support
 * ------------------------------------------------------------ */
%include <rubyopers.swg>

/* ------------------------------------------------------------
 * Warnings for Ruby keywords 
 * ------------------------------------------------------------ */
%include <rubykw.swg>

/* ------------------------------------------------------------
 * Documentation for common Ruby methods 
 * ------------------------------------------------------------ */
%include <rubyautodoc.swg>

/* ------------------------------------------------------------
 * The Ruby initialization function 
 * ------------------------------------------------------------ */
%include <rubyinit.swg>



