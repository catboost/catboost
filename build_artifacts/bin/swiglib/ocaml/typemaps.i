/* ----------------------------------------------------------------------------
 * typemaps.i
 *
 * These typemaps provide support for input/output arguments for C/C++ pointers
 * and C++ references.
* ---------------------------------------------------------------------------- */

%define INPUT_OUTPUT_INOUT_TYPEMAPS(type, c_to_ocaml, ocaml_to_c)
%typemap(in) type *INPUT(type temp), type &INPUT(type temp) {
  temp = (type)ocaml_to_c($input);
  $1 = &temp;
}
%typemap(typecheck) type *INPUT = type;
%typemap(typecheck) type &INPUT = type;

%typemap(in, numinputs=0) type *OUTPUT($*1_ltype temp), type &OUTPUT($*1_ltype temp) "$1 = &temp;"
%typemap(argout) type *OUTPUT, type &OUTPUT {
  swig_result = caml_list_append(swig_result, c_to_ocaml(*$1));
}
%typemap(in) type *INOUT = type *INPUT;
%typemap(in) type &INOUT = type &INPUT;

%typemap(argout) type *INOUT = type *OUTPUT;
%typemap(argout) type &INOUT = type &OUTPUT;

%typemap(typecheck) type *INOUT = type;
%typemap(typecheck) type &INOUT = type;
%enddef

INPUT_OUTPUT_INOUT_TYPEMAPS(bool, caml_val_bool, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(int, caml_val_int, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(long, caml_val_long, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(short, caml_val_int, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(char, caml_val_char, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(signed char, caml_val_char, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(float, caml_val_float, caml_double_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(double, caml_val_double, caml_double_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(unsigned int, caml_val_uint, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(unsigned long, caml_val_ulong, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(unsigned short, caml_val_ushort, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(unsigned char, caml_val_uchar, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(long long, caml_val_long, caml_long_val);
INPUT_OUTPUT_INOUT_TYPEMAPS(unsigned long long, caml_val_ulong, caml_long_val);
#undef INPUT_OUTPUT_INOUT_TYPEMAPS
