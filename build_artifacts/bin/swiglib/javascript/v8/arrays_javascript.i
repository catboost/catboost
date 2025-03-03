/* -----------------------------------------------------------------------------
 * arrays_javascript.i
 *
 * These typemaps give more natural support for arrays. The typemaps are not efficient
 * as there is a lot of copying of the array values whenever the array is passed to C/C++
 * from JavaScript and vice versa. The JavaScript array is expected to be the same size as the C array.
 * An exception is thrown if they are not.
 *
 * Example usage:
 * Wrapping:
 *
 *   %include <arrays_javascript.i>
 *   %inline %{
 *       extern int FiddleSticks[3];
 *   %}
 *
 * Use from JavaScript like this:
 *
 *   var fs = [10, 11, 12];
 *   example.FiddleSticks = fs;
 *   fs = example.FiddleSticks;
 * ----------------------------------------------------------------------------- */

%fragment("SWIG_JSCGetIntProperty",    "header", fragment=SWIG_AsVal_frag(int)) {}
%fragment("SWIG_JSCGetNumberProperty", "header", fragment=SWIG_AsVal_frag(double)) {}

%typemap(in, fragment="SWIG_JSCGetIntProperty") int[], int[ANY]
    (int length = 0, v8::Local<v8::Array> array, v8::Local<v8::Value> jsvalue, int i = 0, int res = 0, $*1_ltype temp) {
  if ($input->IsArray())
  {
    // Convert into Array
    array = v8::Local<v8::Array>::Cast($input);

    length = $1_dim0;

    $1  = ($*1_ltype *)malloc(sizeof($*1_ltype) * length);

    // Get each element from array
    for (i = 0; i < length; i++)
    {
      jsvalue = array->Get(i);

      // Get primitive value from JSObject
      res = SWIG_AsVal(int)(jsvalue, &temp);
      if (!SWIG_IsOK(res))
      {
        SWIG_exception_fail(SWIG_ERROR, "Failed to convert $input to double");
      }
      arg$argnum[i] = temp;
    }

  }
  else
  {
    SWIG_exception_fail(SWIG_ERROR, "$input is not JSObjectRef");
  }
}

%typemap(freearg) int[], int[ANY] {
    free($1);
}

%typemap(out, fragment=SWIG_From_frag(int)) int[], int[ANY] (int length = 0, int i = 0)
{
  length = $1_dim0;
  v8::Local<v8::Array> array = v8::Array::New(length);

  for (i = 0; i < length; i++)
  {
    array->Set(i, SWIG_From(int)($1[i]));
  }


  $result = array;
}

%typemap(in, fragment="SWIG_JSCGetNumberProperty") double[], double[ANY]
    (int length = 0, v8::Local<v8::Array> array, v8::Local<v8::Value> jsvalue, int i = 0, int res = 0, $*1_ltype temp) {
  if ($input->IsArray())
  {
    // Convert into Array
    array = v8::Local<v8::Array>::Cast($input);

    length = $1_dim0;

    $1  = ($*1_ltype *)malloc(sizeof($*1_ltype) * length);

    // Get each element from array
    for (i = 0; i < length; i++)
    {
      jsvalue = array->Get(i);

      // Get primitive value from JSObject
      res = SWIG_AsVal(double)(jsvalue, &temp);
      if (!SWIG_IsOK(res))
      {
        SWIG_exception_fail(SWIG_ERROR, "Failed to convert $input to double");
      }
      arg$argnum[i] = temp;
    }

  }
  else
  {
    SWIG_exception_fail(SWIG_ERROR, "$input is not JSObjectRef");
  }
}

%typemap(freearg) double[], double[ANY] {
    free($1);
}

%typemap(out, fragment=SWIG_From_frag(double)) double[], double[ANY] (int length = 0, int i = 0)
{
  length = $1_dim0;
  v8::Local<v8::Array> array = v8::Array::New(length);

  for (i = 0; i < length; i++)
  {
    array->Set(i, SWIG_From(double)($1[i]));
  }


  $result = array;
}
