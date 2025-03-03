/* -----------------------------------------------------------------------------
 * std_wstring.i
 *
 * Typemaps for std::wstring and const std::wstring&
 *
 * These are mapped to a Java String and are passed around by value.
 * Warning: Unicode / multibyte characters are handled differently on different 
 * OSs so the std::wstring typemaps may not always work as intended.
 *
 * To use non-const std::wstring references use the following %apply.  Note 
 * that they are passed by value.
 * %apply const std::wstring & {std::wstring &};
 * ----------------------------------------------------------------------------- */

namespace std {

%naturalvar wstring;

class wstring;

// wstring
%typemap(jni) wstring "jstring"
%typemap(jtype) wstring "String"
%typemap(jstype) wstring "String"
%typemap(javadirectorin) wstring "$jniinput"
%typemap(javadirectorout) wstring "$javacall"

%typemap(in) wstring
%{if(!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null std::wstring");
    return $null;
  }
  const jchar *$1_pstr = jenv->GetStringChars($input, 0);
  if (!$1_pstr) return $null;
  jsize $1_len = jenv->GetStringLength($input);
  if ($1_len) {
    $1.reserve($1_len);
    for (jsize i = 0; i < $1_len; ++i) {
      $1.push_back((wchar_t)$1_pstr[i]);
    }
  }
  jenv->ReleaseStringChars($input, $1_pstr);
 %}

%typemap(directorout) wstring
%{if(!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null std::wstring");
    return $null;
  }
  const jchar *$1_pstr = jenv->GetStringChars($input, 0);
  if (!$1_pstr) return $null;
  jsize $1_len = jenv->GetStringLength($input);
  if ($1_len) {
    $result.reserve($1_len);
    for (jsize i = 0; i < $1_len; ++i) {
      $result.push_back((wchar_t)$1_pstr[i]);
    }
  }
  jenv->ReleaseStringChars($input, $1_pstr);
 %}

%typemap(directorin,descriptor="Ljava/lang/String;") wstring %{
  jsize $1_len = $1.length();
  jchar *$1_conv_buf = new jchar[$1_len];
  for (jsize i = 0; i < $1_len; ++i) {
    $1_conv_buf[i] = (jchar)$1[i];
  }
  $input = jenv->NewString($1_conv_buf, $1_len);
  Swig::LocalRefGuard $1_refguard(jenv, $input);
  delete [] $1_conv_buf;
%}

%typemap(out) wstring
%{jsize $1_len = $1.length();
  jchar *conv_buf = new jchar[$1_len];
  for (jsize i = 0; i < $1_len; ++i) {
    conv_buf[i] = (jchar)$1[i];
  }
  $result = jenv->NewString(conv_buf, $1_len);
  delete [] conv_buf; %}

%typemap(javain) wstring "$javainput"

%typemap(javaout) wstring {
    return $jnicall;
  }

//%typemap(typecheck) wstring = wchar_t *;

%typemap(throws) wstring
%{ std::string message($1.begin(), $1.end());
   SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, message.c_str());
   return $null; %}

// const wstring &
%typemap(jni) const wstring & "jstring"
%typemap(jtype) const wstring & "String"
%typemap(jstype) const wstring & "String"
%typemap(javadirectorin) const wstring & "$jniinput"
%typemap(javadirectorout) const wstring & "$javacall"

%typemap(in) const wstring &
%{if(!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null std::wstring");
    return $null;
  }
  const jchar *$1_pstr = jenv->GetStringChars($input, 0);
  if (!$1_pstr) return $null;
  jsize $1_len = jenv->GetStringLength($input);
  std::wstring $1_str;
  if ($1_len) {
    $1_str.reserve($1_len);
    for (jsize i = 0; i < $1_len; ++i) {
      $1_str.push_back((wchar_t)$1_pstr[i]);
    }
  }
  $1 = &$1_str;
  jenv->ReleaseStringChars($input, $1_pstr);
 %}

%typemap(directorout,warning=SWIGWARN_TYPEMAP_THREAD_UNSAFE_MSG) const wstring & 
%{if(!$input) {
    SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null std::wstring");
    return $null;
  }
  const jchar *$1_pstr = jenv->GetStringChars($input, 0);
  if (!$1_pstr) return $null;
  jsize $1_len = jenv->GetStringLength($input);
  /* possible thread/reentrant code problem */
  static std::wstring $1_str;
  if ($1_len) {
    $1_str.reserve($1_len);
    for (jsize i = 0; i < $1_len; ++i) {
      $1_str.push_back((wchar_t)$1_pstr[i]);
    }
  }
  $result = &$1_str;
  jenv->ReleaseStringChars($input, $1_pstr); %}

%typemap(directorin,descriptor="Ljava/lang/String;") const wstring & %{
  jsize $1_len = $1.length();
  jchar *$1_conv_buf = new jchar[$1_len];
  for (jsize i = 0; i < $1_len; ++i) {
    $1_conv_buf[i] = (jchar)($1)[i];
  }
  $input = jenv->NewString($1_conv_buf, $1_len);
  Swig::LocalRefGuard $1_refguard(jenv, $input);
  delete [] $1_conv_buf;
%}

%typemap(out) const wstring & 
%{jsize $1_len = $1->length();
  jchar *conv_buf = new jchar[$1_len];
  for (jsize i = 0; i < $1_len; ++i) {
    conv_buf[i] = (jchar)(*$1)[i];
  }
  $result = jenv->NewString(conv_buf, $1_len);
  delete [] conv_buf; %}

%typemap(javain) const wstring & "$javainput"

%typemap(javaout) const wstring & {
    return $jnicall;
  }

//%typemap(typecheck) const wstring & = wchar_t *;

%typemap(throws) const wstring &
%{ std::string message($1.begin(), $1.end());
   SWIG_JavaThrowException(jenv, SWIG_JavaRuntimeException, message.c_str());
   return $null; %}

}

