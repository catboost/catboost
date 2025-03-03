/*
    The typemaps here allow to handle functions returning std::auto_ptr<>,
    which is the most common use of this type. If you have functions taking it
    as parameter, these typemaps can't be used for them and you need to do
    something else (e.g. use shared_ptr<> which SWIG supports fully).
 */

%define %auto_ptr(TYPE)
%typemap (jni) std::auto_ptr<TYPE > "jlong"
%typemap (jtype) std::auto_ptr<TYPE > "long"
%typemap (jstype) std::auto_ptr<TYPE > "$typemap(jstype, TYPE)"

%typemap (out) std::auto_ptr<TYPE > %{
   jlong lpp = 0;
   *(TYPE**) &lpp = $1.release();
   $result = lpp;
%}
%typemap(javaout) std::auto_ptr<TYPE > {
     long cPtr = $jnicall;
     return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
   }
%template() std::auto_ptr<TYPE >;
%enddef

namespace std {
   template <class T> class auto_ptr {};
} 
