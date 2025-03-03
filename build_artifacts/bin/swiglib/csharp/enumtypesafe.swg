/* -----------------------------------------------------------------------------
 * enumtypesafe.swg
 *
 * Include this file in order for C/C++ enums to be wrapped by the so called
 * typesafe enum pattern. Each enum has an equivalent C# class named after the
 * enum and each enum item is a static instance of this class.
 * ----------------------------------------------------------------------------- */

// const enum SWIGTYPE & typemaps
%typemap(ctype) const enum SWIGTYPE & "int"
%typemap(imtype) const enum SWIGTYPE & "int"
%typemap(cstype) const enum SWIGTYPE & "$*csclassname"

%typemap(in) const enum SWIGTYPE & ($*1_ltype temp)
%{ temp = ($*1_ltype)$input; 
   $1 = &temp; %}
%typemap(out) const enum SWIGTYPE & %{ $result = (int)*$1; %}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const enum SWIGTYPE &
%{ static $*1_ltype temp = ($*1_ltype)$input; 
   $result = &temp; %}
%typemap(directorin) const enum SWIGTYPE & "$input = (int)$1;"
%typemap(csdirectorin) const enum SWIGTYPE & "$*csclassname.swigToEnum($iminput)"
%typemap(csdirectorout) const enum SWIGTYPE & "$cscall.swigValue"

%typecheck(SWIG_TYPECHECK_POINTER) const enum SWIGTYPE & ""

%typemap(throws, canthrow=1) const enum SWIGTYPE &
%{ (void)$1;
   SWIG_CSharpSetPendingException(SWIG_CSharpApplicationException, "C++ $1_type exception thrown");
   return $null; %}

%typemap(csin) const enum SWIGTYPE & "$csinput.swigValue"
%typemap(csout, excode=SWIGEXCODE) const enum SWIGTYPE & {
    $*csclassname ret = $*csclassname.swigToEnum($imcall);$excode
    return ret;
  }

%typemap(csvarout, excode=SWIGEXCODE2) const enum SWIGTYPE & %{
    get {
      $*csclassname ret = $*csclassname.swigToEnum($imcall);$excode
      return ret;
    } %}


// enum SWIGTYPE typemaps
%typemap(ctype) enum SWIGTYPE "int"
%typemap(imtype) enum SWIGTYPE "int"
%typemap(cstype) enum SWIGTYPE "$csclassname"

%typemap(in) enum SWIGTYPE %{ $1 = ($1_ltype)$input; %}
%typemap(out) enum SWIGTYPE %{ $result = (int)$1; %}

%typemap(directorout) enum SWIGTYPE  %{ $result = ($1_ltype)$input; %}
%typemap(directorin) enum SWIGTYPE "$input = (int)$1;"
%typemap(csdirectorin) enum SWIGTYPE "$csclassname.swigToEnum($iminput)"
%typemap(csdirectorout) enum SWIGTYPE "$cscall.swigValue"

%typecheck(SWIG_TYPECHECK_POINTER) enum SWIGTYPE ""

%typemap(throws, canthrow=1) enum SWIGTYPE
%{ (void)$1;
   SWIG_CSharpSetPendingException(SWIG_CSharpApplicationException, "C++ $1_type exception thrown");
   return $null; %}

%typemap(csin) enum SWIGTYPE "$csinput.swigValue"
%typemap(csout, excode=SWIGEXCODE) enum SWIGTYPE {
    $csclassname ret = $csclassname.swigToEnum($imcall);$excode
    return ret;
  }

%typemap(csvarout, excode=SWIGEXCODE2) enum SWIGTYPE %{
    get {
      $csclassname ret = $csclassname.swigToEnum($imcall);$excode
      return ret;
    } %}

%typemap(csbase)           enum SWIGTYPE ""
%typemap(csclassmodifiers) enum SWIGTYPE "public sealed class"
%typemap(cscode)           enum SWIGTYPE ""
%typemap(csimports)        enum SWIGTYPE ""
%typemap(csinterfaces)     enum SWIGTYPE ""

/*
 * The swigToEnum method is used to find the C# enum from a C++ enum integer value. The default one here takes 
 * advantage of the fact that most enums do not have initial values specified, so the lookup is fast. If initial
 * values are specified then a lengthy linear search through all possible enums might occur. Specific typemaps could be
 * written to possibly optimise this lookup by taking advantage of characteristics peculiar to the targeted enum.
 * The special variable, $enumvalues, is replaced with a comma separated list of all the enum values.
 */
%typemap(csbody) enum SWIGTYPE %{
  public readonly int swigValue;

  public static $csclassname swigToEnum(int swigValue) {
    if (swigValue < swigValues.Length && swigValue >= 0 && swigValues[swigValue].swigValue == swigValue)
      return swigValues[swigValue];
    for (int i = 0; i < swigValues.Length; i++)
      if (swigValues[i].swigValue == swigValue)
        return swigValues[i];
    throw new global::System.ArgumentOutOfRangeException("No enum $csclassname with value " + swigValue);
  }

  public override string ToString() {
    return swigName;
  }

  private $csclassname(string swigName) {
    this.swigName = swigName;
    this.swigValue = swigNext++;
  }

  private $csclassname(string swigName, int swigValue) {
    this.swigName = swigName;
    this.swigValue = swigValue;
    swigNext = swigValue+1;
  }

  private $csclassname(string swigName, $csclassname swigEnum) {
    this.swigName = swigName;
    this.swigValue = swigEnum.swigValue;
    swigNext = this.swigValue+1;
  }

  private static $csclassname[] swigValues = { $enumvalues };
  private static int swigNext = 0;
  private readonly string swigName;
%}

%csenum(typesafe);

