
// Include fundamental fragment definitions
%include <typemaps/fragments.swg>

// Look for user fragments file.
%include <octfragments.swg>

// Octave fragments for primitive types
%include <octprimtypes.swg>

// Octave fragments for char* strings
//%include <octstrings.swg>


#ifndef SWIG_DIRECTOR_TYPEMAPS
#define SWIG_DIRECTOR_TYPEMAPS
#endif

// Octave types
#define SWIG_Object                      octave_value
#define VOID_Object                      octave_value()

/*
// Octave allows implicit conversion
#define %implicitconv_flag              $implicitconv 
*/

// append output
#define SWIG_AppendOutput(result, obj)  SWIG_Octave_AppendOutput(result, obj)

// set constant
#define SWIG_SetConstant(name, obj)     SWIG_Octave_SetConstant(module_ns,name,obj) 

// raise
#define SWIG_Octave_Raise(OBJ, TYPE, DESC) error("C++ side threw an exception of type " TYPE)
#define SWIG_Raise(obj, type, desc)     SWIG_Octave_Raise(obj, type, desc)

// Include the unified typemap library
%include <typemaps/swigtypemaps.swg>

%typecheck(SWIG_TYPECHECK_SWIGOBJECT) SWIG_Object "$1 = (*$input).is_defined();";
%typecheck(SWIG_TYPECHECK_SWIGOBJECT) octave_value_list "$1 = true;";

%typemap(in) (octave_value_list varargs,...) {
  for (int j=$argnum-1;j<args.length();++j)
    $1.append(args(j));
}
%typecheck(2000) (octave_value_list varargs,...) {
  $1=1;
}

%typemap(in) (const octave_value_list& varargs,...) (octave_value_list tmp) {
  for (int j=$argnum-1;j<args.length();++j)
    tmp.append(args(j));
  $1=&tmp;
}
%typecheck(2000) (const octave_value_list& varargs,...) {
  $1=1;
}

%typemap(out) octave_value_list {
  _outp->append($1);
}
%typemap(out,noblock=1) octave_map, Octave_map {
  $result=$1;
}
%typemap(out,noblock=1) NDArray {
  $result=$1;
}
%typemap(out,noblock=1) Cell {
  $result=$1;
}

/*
// Smart Pointers
%typemap(out,noblock=1) const SWIGTYPE & SMARTPOINTER  {
  $result = SWIG_NewPointerObj(%new_copy(*$1, $*ltype), $descriptor, SWIG_POINTER_OWN | %newpointer_flags);
}

%typemap(ret) const SWIGTYPE & SMARTPOINTER, SWIGTYPE SMARTPOINTER {
  octave_swig_type* lobj=Swig::swig_value_deref($result);
  if (lobj) {
    std::list<octave_value_list> idx;
    idx.push_back(octave_value("__deref__"));
    idx.push_back(octave_value_list());
    octave_value_list ovl(lobj->subsref(".(",idx));
    octave_swig_type* robj=ovl.length()>=1?Swig::swig_value_deref(ovl(0)):0;
    if (robj && !error_state)
      lobj->append(robj);
  }
}
*/
