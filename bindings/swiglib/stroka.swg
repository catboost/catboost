%header %{
#include <util/generic/string.h>
#include <util/charset/wide.h>
%}
%include <typemaps.i>

#if defined(SWIGJAVA)
%naturalvar TString;

%typemap(jni) TString, const TString & "jstring"
%typemap(jtype) TString, const TString & "String"
%typemap(jstype) TString, const TString & "String"

%typemap(in) TString {
    if(!$input) {
        SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null TString");
        return $null;
    }
    const char *$1_pstr = (const char *)jenv->GetStringUTFChars($input, 0);
    if (!$1_pstr) return $null;
    jsize $1_len = jenv->GetStringUTFLength($input);
    if ($1_len) {
        $1.assign($1_pstr, $1_len);
    }
    jenv->ReleaseStringUTFChars($input, $1_pstr);
}

%typemap(in) const TString & ( TString temp ) {
    if(!$input) {
        SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null TString");
        return $null;
    }
    const char *$1_pstr = (const char *)jenv->GetStringUTFChars($input, 0);
    if (!$1_pstr) return $null;
    jsize $1_len = jenv->GetStringUTFLength($input);
    if ($1_len) {
        temp.assign($1_pstr, $1_len);
    }
    $1 = &temp;
    jenv->ReleaseStringUTFChars($input, $1_pstr);
}

%typemap(out) TString {
    $result = jenv->NewStringUTF($1.data());
}

%typemap(out) TString *, TString & {
    $result = jenv->NewStringUTF($1->data());
}

%typemap(javain) TString, const TString & "$javainput"

%typemap(javaout) TString, const TString & {
    return $jnicall;
}

#else
%include <typemaps/std_strings.swg>
%typemaps_std_string(TString, char, SWIG_AsCharPtrAndSize, SWIG_FromCharPtrAndSize, %checkcode(STDSTRING));
#endif

