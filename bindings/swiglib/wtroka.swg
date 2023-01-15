%header %{
#include <library/cpp/charset/wide.h>
#include <util/generic/string.h>
#include <util/charset/wide.h>
%}
#%include <typemaps.i>

%naturalvar TUtf16String;

#if defined(SWIGPYTHON)
%typemap(in) TUtf16String {
    if (PyUnicode_Check($input)) {
        PyObject* u = PyUnicode_AsEncodedString($input, "UTF-16LE", 0);
        $1.assign((wchar16*)PyBytes_AS_STRING(u), (size_t)PyBytes_GET_SIZE(u) / 2);
        Py_XDECREF(u);
    } else {
        %argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
    }
}

%typemap(in) const TUtf16String& ( TUtf16String temp ) {
    if (PyUnicode_Check($input)) {
        PyObject* u = PyUnicode_AsEncodedString($input, "UTF-16LE", 0);
        temp.assign((wchar16*)PyBytes_AS_STRING(u), (size_t)PyBytes_GET_SIZE(u) / 2);
        $1 = &temp;
        Py_XDECREF(u);
    } else {
        %argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
    }
}

%typemap(out) TUtf16String {
    int byteorder = 0;
    $result = PyUnicode_DecodeUTF16((const char *)($1).data(), (int)($1).size() * 2, 0, &byteorder);
}

%typemap(out) TUtf16String *, TUtf16String & {
    int byteorder = 0;
    $result = PyUnicode_DecodeUTF16((const char *)(*$1).data(), (int)(*$1).size() * 2, 0, &byteorder);
}

#elif defined(SWIGPERL)
%typemap(in) TUtf16String {
    if (SvMAGICAL($input)) {
        SV *tmp = sv_newmortal();
        SvSetSV(tmp, $input);
        $input = tmp;
    }
    if (SvPOK($input)) {
        char *cstr = SvPV_nolen($input);
        bool isUtf = SvUTF8($input);
        $1 = CharToWide<true>(cstr, sv_len($input), isUtf ? CODES_UTF8 : CODES_YANDEX);
    } else {
        %argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
    }
}

%typemap(in) const TUtf16String& ( TUtf16String temp ) {
    if (SvMAGICAL($input)) {
        SV *tmp = sv_newmortal();
        SvSetSV(tmp, $input);
        $input = tmp;
    }
    if (SvPOK($input)) {
        const char *cstr = SvPV_nolen($input);
        bool isUtf = SvUTF8($input);
        temp = CharToWide<true>(cstr, sv_len($input), isUtf ? CODES_UTF8 : CODES_YANDEX);
        $1 = &temp;
    } else {
        %argument_fail(SWIG_TypeError, "$type", $symname, $argnum);
    }
}

%typemap(out) TUtf16String {
    $result = sv_newmortal();
    TString s = WideToUTF8($1.data(), $1.size());
    sv_setpvn($result, s.data(), s.size());
    SvUTF8_on($result);
    argvi++;
}

%typemap(out) TUtf16String *, TUtf16String &  {
    $result = sv_newmortal();
    TString s = WideToUTF8($1->data(), $1->size());
    sv_setpvn($result, s.data(), s.size());
    SvUTF8_on($result);
    argvi++;
}
#elif defined(SWIGJAVA)
%typemap(jni) TUtf16String, const TUtf16String & "jstring"
%typemap(jtype) TUtf16String, const TUtf16String & "String"
%typemap(jstype) TUtf16String, const TUtf16String & "String"

%typemap(in) TUtf16String {
    if(!$input) {
        SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null TUtf16String");
        return $null;
    }
    const jchar *$1_pstr = jenv->GetStringChars($input, 0);
    if (!$1_pstr) return $null;
    jsize $1_len = jenv->GetStringLength($input);
    if ($1_len) {
        $1.assign($1_pstr, $1_len);
    }
    jenv->ReleaseStringChars($input, $1_pstr);
}

%typemap(in) const TUtf16String & ( TUtf16String temp ) {
    if(!$input) {
        SWIG_JavaThrowException(jenv, SWIG_JavaNullPointerException, "null TUtf16String");
        return $null;
    }
    const jchar *$1_pstr = jenv->GetStringChars($input, 0);
    if (!$1_pstr) return $null;
    jsize $1_len = jenv->GetStringLength($input);
    if ($1_len) {
        temp.assign($1_pstr, $1_len);
    }
    $1 = &temp;
    jenv->ReleaseStringChars($input, $1_pstr);
}

%typemap(out) TUtf16String {
    $result = jenv->NewString((jchar*)$1.data(), $1.size());
}

%typemap(out) TUtf16String *, TUtf16String & {
    $result = jenv->NewString((jchar*)$1->data(), $1->size());
}

%typemap(javain) TUtf16String, const TUtf16String & "$javainput"

%typemap(javaout) TUtf16String, const TUtf16String & {
    return $jnicall;
}

#else
#warning "No typemaps for TUtf16String are defined!"
#endif

