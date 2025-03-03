// Users can provide their own SWIG_INTRUSIVE_PTR_TYPEMAPS or SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP macros before including this file to change the
// visibility of the constructor and getCPtr method if desired to public if using multiple modules.
#ifndef SWIG_INTRUSIVE_PTR_TYPEMAPS
#define SWIG_INTRUSIVE_PTR_TYPEMAPS(CONST, TYPE...) SWIG_INTRUSIVE_PTR_TYPEMAPS_IMPLEMENTATION(internal, internal, CONST, TYPE)
#endif
#ifndef SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP
#define SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP(CONST, TYPE...) SWIG_INTRUSIVE_PTR_TYPEMAPS_NO_WRAP_IMPLEMENTATION(internal, internal, CONST, TYPE)
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

%typemap(in, canthrow=1) CONST TYPE ($&1_type argp = 0) %{
  // plain value
  argp = (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0;
  if (!argp) {
    SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentNullException, "Attempt to dereference null $1_type", 0);
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

%typemap(in, canthrow=1) CONST TYPE * (SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > *smartarg = 0) %{
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

%typemap(in, canthrow=1) CONST TYPE & %{
  // plain reference
  $1 = ($1_ltype)((*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0);
  if(!$1) {
    SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentNullException, "$1_type reference is null", 0);
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

%typemap(in) TYPE *CONST& ($*1_ltype temp = 0) %{
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

%typemap(in) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > (SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > * smartarg) %{
  // intrusive_ptr by value
  smartarg = *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >**)&$input;
  if (smartarg) {
    $1 = SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >(smartarg->get(), true);
  }
%}
%typemap(out, fragment="SWIG_intrusive_deleter") SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > %{
  if ($1) {
    intrusive_ptr_add_ref($1.get());
    *(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$result = new SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >($1.get(), SWIG_intrusive_deleter< CONST TYPE >());
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


%typemap (ctype)    SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "void *"
%typemap (imtype, out="global::System.IntPtr")  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "global::System.Runtime.InteropServices.HandleRef"
%typemap (cstype) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                  SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "$typemap(cstype, TYPE)"
%typemap(csin) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >,
                 SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE >,
                 SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > &,
                 SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *,
                 SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& "$typemap(cstype, TYPE).getCPtr($csinput)"

%typemap(csout, excode=SWIGEXCODE) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > {
    global::System.IntPtr cPtr = $imcall;
    $typemap(cstype, TYPE) ret = (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > {
    global::System.IntPtr cPtr = $imcall;
    $typemap(cstype, TYPE) ret = (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > & {
    global::System.IntPtr cPtr = $imcall;
    $typemap(cstype, TYPE) ret = (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > * {
    global::System.IntPtr cPtr = $imcall;
    $typemap(cstype, TYPE) ret = (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > *& {
    global::System.IntPtr cPtr = $imcall;
    $typemap(cstype, TYPE) ret = (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
    return ret;
  }
%typemap(csvarout, excode=SWIGEXCODE2) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE > %{
    get {
      $typemap(cstype, TYPE) ret = new $typemap(cstype, TYPE)($imcall, true);$excode
      return ret;
    } %}
%typemap(csvarout, excode=SWIGEXCODE2) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >& %{
    get {
      $typemap(cstype, TYPE) ret = new $typemap(cstype, TYPE)($imcall, true);$excode
      return ret;
    } %}
%typemap(csvarout, excode=SWIGEXCODE2) SWIG_INTRUSIVE_PTR_QNAMESPACE::intrusive_ptr< CONST TYPE >* %{
    get {
      $typemap(cstype, TYPE) ret = new $typemap(cstype, TYPE)($imcall, true);$excode
      return ret;
    } %}


%typemap(csout, excode=SWIGEXCODE) CONST TYPE {
    $typemap(cstype, TYPE) ret = new $typemap(cstype, TYPE)($imcall, true);$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) CONST TYPE & {
    $typemap(cstype, TYPE) ret = new $typemap(cstype, TYPE)($imcall, true);$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) CONST TYPE * {
    global::System.IntPtr cPtr = $imcall;
    $typemap(cstype, TYPE) ret = (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
    return ret;
  }
%typemap(csout, excode=SWIGEXCODE) TYPE *CONST& {
    global::System.IntPtr cPtr = $imcall;
    $typemap(cstype, TYPE) ret = (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);$excode
    return ret;
  }

// Base proxy classes
%typemap(csbody) TYPE %{
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  private bool swigCMemOwnBase;

  PTRCTOR_VISIBILITY $csclassname(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwnBase = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  CPTR_VISIBILITY static global::System.Runtime.InteropServices.HandleRef getCPtr($csclassname obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }
%}

// Derived proxy classes
%typemap(csbody_derived) TYPE %{
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  private bool swigCMemOwnDerived;

  PTRCTOR_VISIBILITY $csclassname(global::System.IntPtr cPtr, bool cMemoryOwn) : base($imclassname.$csclazznameSWIGSmartPtrUpcast(cPtr), true) {
    swigCMemOwnDerived = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  CPTR_VISIBILITY static global::System.Runtime.InteropServices.HandleRef getCPtr($csclassname obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }
%}

%typemap(csdisposing, methodname="Dispose", methodmodifiers="protected", parameters="bool disposing") TYPE {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwnBase) {
          swigCMemOwnBase = false;
          $imcall;
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
    }
  }

%typemap(csdisposing_derived, methodname="Dispose", methodmodifiers="protected", parameters="bool disposing") TYPE {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwnDerived) {
          swigCMemOwnDerived = false;
          $imcall;
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      base.Dispose(disposing);
    }
  }

// CONST version needed ???? also for C#
%typemap(imtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< TYPE > swigSharedPtrUpcast "global::System.Runtime.InteropServices.HandleRef"
%typemap(imtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > swigSharedPtrUpcast "global::System.Runtime.InteropServices.HandleRef"


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
%typemap(in, canthrow=1) CONST TYPE ($&1_type argp = 0) %{
  argp = (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0;
  if (!argp) {
    SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentNullException, "Attempt to dereference null $1_type", 0);
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
%typemap(in, canthrow=1) CONST TYPE & %{
  $1 = ($1_ltype)((*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input) ? (*(SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > **)&$input)->get() : 0);
  if (!$1) {
    SWIG_CSharpSetPendingExceptionArgument(SWIG_CSharpArgumentNullException, "$1_type reference is null", 0);
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


%typemap (ctype)    SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "void *"
%typemap (imtype)  SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "void *"
%typemap (cstype) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "$typemap(cstype, TYPE)"
%typemap (csin) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > "$typemap(cstype, TYPE).getCPtr($csinput)"
%typemap(csout, excode=SWIGEXCODE) SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > {
    global::System.IntPtr cPtr = $imcall;
    return (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);
  }

%typemap(csout, excode=SWIGEXCODE) CONST TYPE {
    return new $typemap(cstype, TYPE)($imcall, true);
  }
%typemap(csout, excode=SWIGEXCODE) CONST TYPE & {
    return new $typemap(cstype, TYPE)($imcall, true);
  }
%typemap(csout, excode=SWIGEXCODE) CONST TYPE * {
    global::System.IntPtr cPtr = $imcall;
    return (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);
  }
%typemap(csout, excode=SWIGEXCODE) TYPE *CONST& {
    global::System.IntPtr cPtr = $imcall;
    return (cPtr == global::System.IntPtr.Zero) ? null : new $typemap(cstype, TYPE)(cPtr, true);
  }

// Base proxy classes
%typemap(csbody) TYPE %{
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  private bool swigCMemOwnBase;

  PTRCTOR_VISIBILITY $csclassname(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwnBase = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  CPTR_VISIBILITY static global::System.Runtime.InteropServices.HandleRef getCPtr($csclassname obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }
%}

// Derived proxy classes
%typemap(csbody_derived) TYPE %{
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  private bool swigCMemOwnDerived;

  PTRCTOR_VISIBILITY $csclassname(global::System.IntPtr cPtr, bool cMemoryOwn) : base($imclassname.$csclazznameSWIGSmartPtrUpcast(cPtr), true) {
    swigCMemOwnDerived = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  CPTR_VISIBILITY static global::System.Runtime.InteropServices.HandleRef getCPtr($csclassname obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }
%}

%typemap(csdisposing, methodname="Dispose", methodmodifiers="protected", parameters="bool disposing") TYPE {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwnBase) {
          swigCMemOwnBase = false;
          $imcall;
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
    }
  }

%typemap(csdisposing_derived, methodname="Dispose", methodmodifiers="protected", parameters="bool disposing") TYPE {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwnDerived) {
          swigCMemOwnDerived = false;
          $imcall;
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      base.Dispose(disposing);
    }
  }


// CONST version needed ???? also for C#
%typemap(imtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< TYPE > swigSharedPtrUpcast "global::System.Runtime.InteropServices.HandleRef"
%typemap(imtype, nopgcpp="1") SWIG_SHARED_PTR_QNAMESPACE::shared_ptr< CONST TYPE > swigSharedPtrUpcast "global::System.Runtime.InteropServices.HandleRef"

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
