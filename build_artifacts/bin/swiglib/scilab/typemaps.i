/* -----------------------------------------------------------------------------
 * typemaps.i
 *
 * ----------------------------------------------------------------------------- */

// INPUT typemaps
%define %scilab_input_typemap(Type)
%typemap(in, noblock=1, fragment=SWIG_AsVal_frag(Type)) Type *INPUT(Type temp)(int ecode), Type &INPUT(Type temp)(int ecode) {
  ecode = SWIG_AsVal_dec(Type)($input, &temp);
  if (!SWIG_IsOK(ecode)) {
    %argument_fail(ecode, "$type", $symname, $argnum);
  }
  $1 = &temp;
}

%typemap(freearg, noblock=1) Type *INPUT, Type &INPUT {
}

%typemap(typecheck) Type *INPUT, Type &INPUT {
}
%enddef

// OUTPUT typemaps
%define %scilab_output_typemap(Type)
%typemap(argout, noblock=1, fragment=SWIG_From_frag(Type)) Type *OUTPUT, Type &OUTPUT {
  %set_output(SWIG_From_dec(Type)(*$1));
}
%enddef

// INOUT typemaps
%define %scilab_inout_typemap(Type)
 %typemap(in) Type *INOUT = Type *INPUT;
 %typemap(in) Type &INOUT = Type &INPUT;
 %typemap(argout) Type *INOUT = Type *OUTPUT;
 %typemap(argout) Type &INOUT = Type &OUTPUT;
%enddef


%define %scilab_inout_typemaps(Type)
  %scilab_input_typemap(%arg(Type))
  %scilab_output_typemap(%arg(Type))
  %scilab_inout_typemap(%arg(Type))
%enddef

%scilab_inout_typemaps(double);
%scilab_inout_typemaps(signed char);
%scilab_inout_typemaps(unsigned char);
%scilab_inout_typemaps(short);
%scilab_inout_typemaps(unsigned short);
%scilab_inout_typemaps(int);
%scilab_inout_typemaps(unsigned int);
%scilab_inout_typemaps(long);
%scilab_inout_typemaps(unsigned long);
%scilab_inout_typemaps(bool);
%scilab_inout_typemaps(float);

//%apply_ctypes(%scilab_inout_typemaps);





