// Users can provide their own SWIG_INTRUSIVE_PTR_TYPEMAPS or SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP macros before including this file to change the
// visibility of the constructor and getCPtr method if desired to public if using multiple modules.
#ifndef SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP
#define SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP(CONST, TYPE...) SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP_IMPLEMENTATION(protected, protected, CONST, TYPE)
#endif
#ifndef SWIG_INTRUSIVE_PTR_TYPEMAPS
#define SWIG_INTRUSIVE_PTR_TYPEMAPS(CONST, TYPE...) SWIG_INTRUSIVE_PTR_TYPEMAPS_IMPLEMENTATION(protected, protected, CONST, TYPE)
#endif


%include <intrusive_ptr.i>

// Language specific macro implementing all the customisations for handling the smart pointer
%define SWIG_INTRUSIVE_PTR_TYPEMAPS_IMPLEMENTATION(PTRCTOR_VISIBILITY, CPTR_VISIBILITY, CONST, TYPE...)

// %naturalvar is as documented for member variables
%naturalvar TYPE;
%naturalvar SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >;

// destructor wrapper customisation
%feature("unref") TYPE "(void)arg1; delete smartarg1;"

// Typemap customisations...

%typemap(in) CONST TYPE ($&1_type argp = 0, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *smartarg = 0) %{
  // plain value
  argp = (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0;
  if (!argp) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Attempt to dereference null $1_type");
    return $null;
  }
  $1 = *argp;
%}
%typemap(out, fragment="SWIG_intrusive_deleter") CONST TYPE %{
  //plain value(out)
  $1_ltype* resultp = new $1_ltype(($1_ltype &)$1);
  intrusive_ptr_add_ref(resultp);
  *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >(resultp, SWIG_intrusive_deleter< CONST TYPE >());
%}

%typemap(in) CONST TYPE * (SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *smartarg = 0) %{
  // plain pointer
  smartarg = *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input;
  $1 = (TYPE *)(smartarg ? smartarg->get() : 0);
%}
%typemap(out, fragment="SWIG_intrusive_deleter,SWIG_null_deleter") CONST TYPE * %{
  //plain pointer(out)
  #if ($owner)
  if ($1) {
    intrusive_ptr_add_ref($1);
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1, SWIG_intrusive_deleter< CONST TYPE >());
  } else {
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = 0;
  }
  #else
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = $1 ? new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1 SWIG_NO_NULL_DELETER_0) : 0;
  #endif
%}

%typemap(in) CONST TYPE & (SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *smartarg = 0) %{
  // plain reference
  $1 = ($1_ltype)((*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0);
  if(!$1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "$1_type reference is null");
    return $null;
  }
%}
%typemap(out, fragment="SWIG_intrusive_deleter,SWIG_null_deleter") CONST TYPE & %{
  //plain reference(out)
  #if ($owner)
  if ($1) {
    intrusive_ptr_add_ref($1);
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1, SWIG_intrusive_deleter< CONST TYPE >());
  } else {
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = 0;
  }
  #else
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = $1 ? new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1 SWIG_NO_NULL_DELETER_0) : 0;
  #endif
%}

%typemap(in) TYPE *CONST& ($*1_ltype temp = 0, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *smartarg = 0) %{
  // plain pointer by reference
  temp = ($*1_ltype)((*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0);
  $1 = &temp;
%}
%typemap(out, fragment="SWIG_intrusive_deleter,SWIG_null_deleter") TYPE *CONST& %{
  // plain pointer by reference(out)
  #if ($owner)
  if (*$1) {
    intrusive_ptr_add_ref(*$1);
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >(*$1, SWIG_intrusive_deleter< CONST TYPE >());
  } else {
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = 0;
  }
  #else
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >(*$1 SWIG_NO_NULL_DELETER_0);
  #endif
%}

%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > ($&1_type argp, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > * smartarg) %{
  // intrusive_ptr by value
  smartarg = *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >**)&$input;
  if (smartarg) {
  	$1 = SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >(smartarg->get(), true);
  }
%}
%typemap(out, fragment="SWIG_intrusive_deleter") SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > %{
  if ($1) {
  	intrusive_ptr_add_ref(result.get());
  	*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >(result.get(), SWIG_intrusive_deleter< CONST TYPE >());
  } else {
   	*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = 0;
  }
%}

%typemap(in) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > swigSharedPtrUpcast ($&1_type smartarg) %{
  // shared_ptr by value
  smartarg = *($&1_ltype*)&$input;
  if (smartarg) $1 = *smartarg;
%}
%typemap(out) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > ANY_TYPE_SWIGSharedPtrUpcast %{
  *($&1_ltype*)&$result = $1 ? new $1_ltype($1) : 0;
%}

%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > & ($*1_ltype tempnull, $*1_ltype temp, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > * smartarg) %{
  // intrusive_ptr by reference
  if ( $input ) {
  	smartarg = *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >**)&$input;
  	temp = SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >(smartarg->get(), true);
  	$1 = &temp;
  } else {
	$1 = &tempnull;
  }
%}
%typemap(memberin) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > & %{
  delete &($1);
  if ($self) {
    SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > * temp = new SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >(*$input);
    $1 = *temp;
  }
%}
%typemap(out, fragment="SWIG_intrusive_deleter") SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > & %{
  if (*$1) {
    intrusive_ptr_add_ref($1->get());
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1->get(), SWIG_intrusive_deleter< CONST TYPE >());
  } else {
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = 0;
  }
%}

%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > * ($*1_ltype tempnull, $*1_ltype temp, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > * smartarg) %{
  // intrusive_ptr by pointer
  if ( $input ) {
  	smartarg = *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >**)&$input;
  	temp = SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >(smartarg->get(), true);
  	$1 = &temp;
  } else {
	$1 = &tempnull;
  }
%}
%typemap(memberin) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > * %{
  delete $1;
  if ($self) $1 = new SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >(*$input);
%}
%typemap(out, fragment="SWIG_intrusive_deleter") SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > * %{
  if ($1 && *$1) {
    intrusive_ptr_add_ref($1->get());
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1->get(), SWIG_intrusive_deleter< CONST TYPE >());
  } else {
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = 0;
  }
  if ($owner) delete $1;
%}

%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& (SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > temp, $*1_ltype tempp = 0, SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > * smartarg) %{
  // intrusive_ptr by pointer reference
  smartarg = *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >**)&$input;
  if ($input) {
    temp = SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >(smartarg->get(), true);
  }
  tempp = &temp;
  $1 = &tempp;
%}
%typemap(memberin) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& %{
  if ($self) $1 = *$input;
%}
%typemap(out, fragment="SWIG_intrusive_deleter") SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& %{
  if (*$1 && **$1) {
    intrusive_ptr_add_ref((*$1)->get());
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >((*$1)->get(), SWIG_intrusive_deleter< CONST TYPE >());
  } else {
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = 0;
  }
%}

// various missing typemaps - If ever used (unlikely) ensure compilation error rather than runtime bug
%typemap(in) CONST TYPE[], CONST TYPE[ANY], CONST TYPE (CLASS::*) %{
#error "typemaps for $1_type not available"
%}
%typemap(out) CONST TYPE[], CONST TYPE[ANY], CONST TYPE (CLASS::*) %{
#error "typemaps for $1_type not available"
%}


%typemap (jni)    SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "jlong"
%typemap (jtype)  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "long"
%typemap (jstype) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "$typemap(jstype, TYPE)"
%typemap(javain) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                 SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                 SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                 SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                 SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "$typemap(jstype, TYPE).getCPtr($javainput)"

%typemap(javaout) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }
%typemap(javaout) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }
%typemap(javaout) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > & {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }
%typemap(javaout) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > * {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }
%typemap(javaout) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }


%typemap(javaout) CONST TYPE {
    return new $typemap(jstype, TYPE)($jnicall, true);
  }
%typemap(javaout) CONST TYPE & {
    return new $typemap(jstype, TYPE)($jnicall, true);
  }
%typemap(javaout) CONST TYPE * {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }
%typemap(javaout) TYPE *CONST& {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }

// Base proxy classes
%typemap(javabody) TYPE %{
  private transient long swigCPtr;
  private transient boolean swigCMemOwnBase;

  PTRCTOR_VISIBILITY $javaclassname(long cPtr, boolean cMemoryOwn) {
    swigCMemOwnBase = cMemoryOwn;
    swigCPtr = cPtr;
  }

  CPTR_VISIBILITY static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }
%}

// Derived proxy classes
%typemap(javabody_derived) TYPE %{
  private transient long swigCPtr;
  private transient boolean swigCMemOwnDerived;

  PTRCTOR_VISIBILITY $javaclassname(long cPtr, boolean cMemoryOwn) {
    super($imclassname.$javaclazznameSWIGSmartPtrUpcast(cPtr), true);
    swigCMemOwnDerived = cMemoryOwn;
    swigCPtr = cPtr;
  }

  CPTR_VISIBILITY static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }
%}

%typemap(javadestruct, methodname="delete", methodmodifiers="public synchronized") TYPE {
    if(swigCPtr != 0 && swigCMemOwnBase) {
      swigCMemOwnBase = false;
      $jnicall;
    }
    swigCPtr = 0;
  }

%typemap(javadestruct_derived, methodname="delete", methodmodifiers="public synchronized") TYPE {
    if(swigCPtr != 0 && swigCMemOwnDerived) {
      swigCMemOwnDerived = false;
      $jnicall;
    }
    swigCPtr = 0;
    super.delete();
  }

// CONST version needed ???? also for C#
%typemap(jtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< TYPE > swigSharedPtrUpcast "long"
%typemap(jtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > swigSharedPtrUpcast "long"


%template() SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >;
%template() SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >;
%enddef


/////////////////////////////////////////////////////////////////////


%include <shared_ptr.i>

%define SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP_IMPLEMENTATION(PTRCTOR_VISIBILITY, CPTR_VISIBILITY, CONST, TYPE...)

%naturalvar TYPE;
%naturalvar SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >;

// destructor mods
%feature("unref") TYPE "(void)arg1; delete smartarg1;"


// plain value
%typemap(in) CONST TYPE ($&1_type argp = 0) %{
  argp = (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0;
  if (!argp) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "Attempt to dereference null $1_type");
    return $null;
  }
  $1 = *argp; %}
%typemap(out) CONST TYPE
%{ *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >(new $1_ltype(($1_ltype &)$1)); %}

// plain pointer
%typemap(in) CONST TYPE * (SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *smartarg = 0) %{
  smartarg = *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input;
  $1 = (TYPE *)(smartarg ? smartarg->get() : 0); %}
%typemap(out, fragment="SWIG_null_deleter") CONST TYPE * %{
  *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = $1 ? new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1 SWIG_NO_NULL_DELETER_$owner) : 0;
%}

// plain reference
%typemap(in) CONST TYPE & %{
  $1 = ($1_ltype)((*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0);
  if (!$1) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "$1_type reference is null");
    return $null;
  } %}
%typemap(out, fragment="SWIG_null_deleter") CONST TYPE &
%{ *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1 SWIG_NO_NULL_DELETER_$owner); %}

// plain pointer by reference
%typemap(in) TYPE *CONST& ($*1_ltype temp = 0)
%{ temp = ($*1_ltype)((*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0);
   $1 = &temp; %}
%typemap(out, fragment="SWIG_null_deleter") TYPE *CONST&
%{ *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >(*$1 SWIG_NO_NULL_DELETER_$owner); %}

%typemap(in) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > swigSharedPtrUpcast ($&1_type smartarg) %{
  // shared_ptr by value
  smartarg = *($&1_ltype*)&$input;
  if (smartarg) $1 = *smartarg;
%}
%typemap(out) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > ANY_TYPE_SWIGSharedPtrUpcast %{
  *($&1_ltype*)&$result = $1 ? new $1_ltype($1) : 0;
%}

// various missing typemaps - If ever used (unlikely) ensure compilation error rather than runtime bug
%typemap(in) CONST TYPE[], CONST TYPE[ANY], CONST TYPE (CLASS::*) %{
#error "typemaps for $1_type not available"
%}
%typemap(out) CONST TYPE[], CONST TYPE[ANY], CONST TYPE (CLASS::*) %{
#error "typemaps for $1_type not available"
%}


%typemap (jni)    SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "jlong"
%typemap (jtype)  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "long"
%typemap (jstype) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "$typemap(jstype, TYPE)"
%typemap (javain) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "$typemap(jstype, TYPE).getCPtr($javainput)"
%typemap(javaout) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }

%typemap(javaout) CONST TYPE {
    return new $typemap(jstype, TYPE)($jnicall, true);
  }
%typemap(javaout) CONST TYPE & {
    return new $typemap(jstype, TYPE)($jnicall, true);
  }
%typemap(javaout) CONST TYPE * {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }
%typemap(javaout) TYPE *CONST& {
    long cPtr = $jnicall;
    return (cPtr == 0) ? null : new $typemap(jstype, TYPE)(cPtr, true);
  }

// Base proxy classes
%typemap(javabody) TYPE %{
  private transient long swigCPtr;
  private transient boolean swigCMemOwnBase;

  PTRCTOR_VISIBILITY $javaclassname(long cPtr, boolean cMemoryOwn) {
    swigCMemOwnBase = cMemoryOwn;
    swigCPtr = cPtr;
  }

  CPTR_VISIBILITY static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }
%}

// Derived proxy classes
%typemap(javabody_derived) TYPE %{
  private transient long swigCPtr;
  private transient boolean swigCMemOwnDerived;

  PTRCTOR_VISIBILITY $javaclassname(long cPtr, boolean cMemoryOwn) {
    super($imclassname.$javaclazznameSWIGSmartPtrUpcast(cPtr), true);
    swigCMemOwnDerived = cMemoryOwn;
    swigCPtr = cPtr;
  }

  CPTR_VISIBILITY static long getCPtr($javaclassname obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }
%}

%typemap(javadestruct, methodname="delete", methodmodifiers="public synchronized") TYPE {
    if (swigCPtr != 0) {
      if (swigCMemOwnBase) {
        swigCMemOwnBase = false;
        $jnicall;
      }
      swigCPtr = 0;
    }
  }

%typemap(javadestruct_derived, methodname="delete", methodmodifiers="public synchronized") TYPE {
    if (swigCPtr != 0) {
      if (swigCMemOwnDerived) {
        swigCMemOwnDerived = false;
        $jnicall;
      }
      swigCPtr = 0;
    }
    super.delete();
  }

// CONST version needed ???? also for C#
%typemap(jtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< TYPE > swigSharedPtrUpcast "long"
%typemap(jtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > swigSharedPtrUpcast "long"

// Typecheck typemaps
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER, equivalent="TYPE *")
  TYPE CONST,
  TYPE CONST &,
  TYPE CONST *,
  TYPE *CONST&,
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > &,
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *,
  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *&
  ""

%template() SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >;
%enddef

