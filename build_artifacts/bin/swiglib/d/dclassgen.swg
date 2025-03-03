/* -----------------------------------------------------------------------------
 * dclassgen.swg
 *
 * Typemaps containing D code used when generating D proxy classes.
 * ----------------------------------------------------------------------------- */

%typemap(dbase)               SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(dclassmodifiers)     SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE [], SWIGTYPE (CLASS::*) "class"
%typemap(dcode)               SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(dimports)            SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(dinterfaces)         SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE [], SWIGTYPE (CLASS::*) ""
%typemap(dinterfaces_derived) SWIGTYPE, SWIGTYPE *, SWIGTYPE &, SWIGTYPE [], SWIGTYPE (CLASS::*) ""

// See <denums.swg>.
%typemap(dclassmodifiers) enum SWIGTYPE "enum"
%typemap(dcode) enum SWIGTYPE ""


/*
 * Proxy classes.
 */

%typemap(dconstructor, excode=SWIGEXCODE,directorconnect="\n  swigDirectorConnect();") SWIGTYPE {
  this($imcall, true);$excode$directorconnect
}

%typemap(ddestructor) SWIGTYPE %{
~this() {
  dispose();
}
%}

// We do not use »override« attribute for generated dispose() methods to stay
// somewhat compatible to Phobos and older Tango versions where Object.dispose()
// does not exist.
%typemap(ddispose, methodname="dispose", methodmodifiers="public", parameters="") SWIGTYPE {
  synchronized(this) {
    if (swigCPtr !is null) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        $imcall;
      }
      swigCPtr = null;
    }
  }
}

%typemap(ddispose_derived, methodname="dispose", methodmodifiers="public", parameters="") SWIGTYPE {
  synchronized(this) {
    if (swigCPtr !is null) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        $imcall;
      }
      swigCPtr = null;
      super.dispose();
    }
  }
}


// Unfortunately, the »package« visibility attribute does not work in D when the
// module in question is in the root package (happens if no -package is specified
// at the SWIG command line), so we are stuck with public visibility for
// swigGetCPtr().
%typemap(dbody) SWIGTYPE %{
private void* swigCPtr;
protected bool swigCMemOwn;

public this(void* cObject, bool ownCObject) {
  swigCPtr = cObject;
  swigCMemOwn = ownCObject;
}

public static void* swigGetCPtr(typeof(this) obj) {
  return (obj is null) ? null : obj.swigCPtr;
}

mixin $imdmodule.SwigOperatorDefinitions;
%}


%typemap(dbody_derived) SWIGTYPE %{
private void* swigCPtr;

public this(void* cObject, bool ownCObject) {
  super($imdmodule.$dclazznameUpcast(cObject), ownCObject);
  swigCPtr = cObject;
}

public static void* swigGetCPtr(typeof(this) obj) {
  return (obj is null) ? null : obj.swigCPtr;
}

mixin $imdmodule.SwigOperatorDefinitions;
%}


/*
 * Type wrapper classes.
 */

%typemap(dbody) SWIGTYPE *, SWIGTYPE &, SWIGTYPE [] %{
private void* swigCPtr;

public this(void* cObject, bool futureUse) {
  swigCPtr = cObject;
}

protected this() {
  swigCPtr = null;
}

public static void* swigGetCPtr(typeof(this) obj) {
  return (obj is null) ? null : obj.swigCPtr;
}

mixin $imdmodule.SwigOperatorDefinitions;
%}


/*
 * Member function pointer wrapper classes (see <dmemberfunctionpointers.swg>).
 */

%typemap(dbody) SWIGTYPE (CLASS::*) %{
private char* swigCPtr;

public this(char* cMemberPtr, bool futureUse) {
  swigCPtr = cMemberPtr;
}

protected this() {
  swigCPtr = null;
}

package static char* swigGetCMemberPtr(typeof(this) obj) {
  return (obj is null) ? null : obj.swigCPtr;
}

mixin $imdmodule.SwigOperatorDefinitions;
%}
