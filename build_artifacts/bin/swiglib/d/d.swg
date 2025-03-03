/* -----------------------------------------------------------------------------
 * d.swg
 *
 * Main library file for the D language module. See the D chapter in the SWIG
 * manual for explanation on the typemaps, pragmas, etc. used.
 * ----------------------------------------------------------------------------- */

// Typemaps for exception handling.
%include <dexception.swg>

// Typemaps for primitive types.
%include <dprimitives.swg>

// Typemaps for non-primitive types (C/C++ classes and structs).
%include <dswigtype.swg>

// Typemaps for enumeration types.
%include <denums.swg>

// Typemaps for member function pointers.
%include <dmemberfunctionpointers.swg>

// Typemaps for wrapping pointers to/arrays of C chars as D strings.
%include <dstrings.swg>

// Typemaps for handling void function return types and empty parameter lists.
%include <dvoid.swg>

// Typemaps containing D code used when generating D proxy classes.
%include <dclassgen.swg>

// Mapping of C++ operator overloading methods to D.
%include <doperators.swg>

// Helper code string and exception handling.
%include <dhead.swg>

// Wrapper loader code for dynamically linking the C wrapper library from the D
// wrapper module.
%include <wrapperloader.swg>

// List of all reserved D keywords.
%include <dkw.swg>

// D-specific directives.
%include <ddirectives.swg>
