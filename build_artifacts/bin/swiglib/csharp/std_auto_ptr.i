/*
    The typemaps here allow to handle functions returning std::auto_ptr<>,
    which is the most common use of this type. If you have functions taking it
    as parameter, these typemaps can't be used for them and you need to do
    something else (e.g. use shared_ptr<> which SWIG supports fully).
 */

%define %auto_ptr(TYPE)
%typemap (ctype) std::auto_ptr<TYPE > "void *"
%typemap (imtype, out="System.IntPtr") std::auto_ptr<TYPE > "HandleRef"
%typemap (cstype) std::auto_ptr<TYPE > "$typemap(cstype, TYPE)"
%typemap (out) std::auto_ptr<TYPE > %{
   $result = (void *)$1.release();
%}
%typemap(csout, excode=SWIGEXCODE) std::auto_ptr<TYPE > {
     System.IntPtr cPtr = $imcall;
     $typemap(cstype, TYPE) ret = (cPtr == System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
     return ret;
   }
%template() std::auto_ptr<TYPE >;
%enddef

namespace std {
   template <class T> class auto_ptr {};
} 
