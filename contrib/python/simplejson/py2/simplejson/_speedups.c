/* -*- mode: C; c-file-style: "python"; c-basic-offset: 4 -*- */
#include "Python.h"
#include "structmember.h"

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromSsize_t PyLong_FromSsize_t
#define PyInt_AsSsize_t PyLong_AsSsize_t
#define PyInt_Check(obj) 0
#define PyInt_CheckExact(obj) 0
#define JSON_UNICHR Py_UCS4
#define JSON_InternFromString PyUnicode_InternFromString
#define PyString_GET_SIZE PyUnicode_GET_LENGTH
#define PY2_UNUSED
#define PY3_UNUSED UNUSED
#else /* PY_MAJOR_VERSION >= 3 */
#define PY2_UNUSED UNUSED
#define PY3_UNUSED
#define PyBytes_Check PyString_Check
#define PyUnicode_READY(obj) 0
#define PyUnicode_KIND(obj) (sizeof(Py_UNICODE))
#define PyUnicode_DATA(obj) ((void *)(PyUnicode_AS_UNICODE(obj)))
#define PyUnicode_READ(kind, data, index) ((JSON_UNICHR)((const Py_UNICODE *)(data))[(index)])
#define PyUnicode_GET_LENGTH PyUnicode_GET_SIZE
#define JSON_UNICHR Py_UNICODE
#define JSON_InternFromString PyString_InternFromString
#endif /* PY_MAJOR_VERSION < 3 */

#if PY_VERSION_HEX < 0x03090000
#if !defined(PyObject_CallNoArgs)
#define PyObject_CallNoArgs(callable) PyObject_CallFunctionObjArgs(callable, NULL);
#endif
#if !defined(PyObject_CallOneArg)
#define PyObject_CallOneArg(callable, arg) PyObject_CallFunctionObjArgs(callable, arg, NULL);
#endif
#endif /* PY_VERSION_HEX < 0x03090000 */

#if PY_VERSION_HEX < 0x02070000
#if !defined(PyOS_string_to_double)
#define PyOS_string_to_double json_PyOS_string_to_double
static double
json_PyOS_string_to_double(const char *s, char **endptr, PyObject *overflow_exception);
static double
json_PyOS_string_to_double(const char *s, char **endptr, PyObject *overflow_exception)
{
    double x;
    assert(endptr == NULL);
    assert(overflow_exception == NULL);
    PyFPE_START_PROTECT("json_PyOS_string_to_double", return -1.0;)
    x = PyOS_ascii_atof(s);
    PyFPE_END_PROTECT(x)
    return x;
}
#endif
#endif /* PY_VERSION_HEX < 0x02070000 */

#if PY_VERSION_HEX < 0x02060000
#if !defined(Py_TYPE)
#define Py_TYPE(ob)     (((PyObject*)(ob))->ob_type)
#endif
#if !defined(Py_SIZE)
#define Py_SIZE(ob)     (((PyVarObject*)(ob))->ob_size)
#endif
#if !defined(PyVarObject_HEAD_INIT)
#define PyVarObject_HEAD_INIT(type, size) PyObject_HEAD_INIT(type) size,
#endif
#endif /* PY_VERSION_HEX < 0x02060000 */

#ifdef __GNUC__
#define UNUSED __attribute__((__unused__))
#else
#define UNUSED
#endif

#define DEFAULT_ENCODING "utf-8"

#define PyScanner_Check(op) PyObject_TypeCheck(op, &PyScannerType)
#define PyScanner_CheckExact(op) (Py_TYPE(op) == &PyScannerType)
#define PyEncoder_Check(op) PyObject_TypeCheck(op, &PyEncoderType)
#define PyEncoder_CheckExact(op) (Py_TYPE(op) == &PyEncoderType)

#define JSON_ALLOW_NAN 1
#define JSON_IGNORE_NAN 2

static PyObject *JSON_Infinity = NULL;
static PyObject *JSON_NegInfinity = NULL;
static PyObject *JSON_NaN = NULL;
static PyObject *JSON_EmptyUnicode = NULL;
#if PY_MAJOR_VERSION < 3
static PyObject *JSON_EmptyStr = NULL;
#endif

static PyTypeObject PyScannerType;
static PyTypeObject PyEncoderType;

typedef struct {
    PyObject *large_strings;  /* A list of previously accumulated large strings */
    PyObject *small_strings;  /* Pending small strings */
} JSON_Accu;

static int
JSON_Accu_Init(JSON_Accu *acc);
static int
JSON_Accu_Accumulate(JSON_Accu *acc, PyObject *unicode);
static PyObject *
JSON_Accu_FinishAsList(JSON_Accu *acc);
static void
JSON_Accu_Destroy(JSON_Accu *acc);

#define ERR_EXPECTING_VALUE "Expecting value"
#define ERR_ARRAY_DELIMITER "Expecting ',' delimiter or ']'"
#define ERR_ARRAY_VALUE_FIRST "Expecting value or ']'"
#define ERR_OBJECT_DELIMITER "Expecting ',' delimiter or '}'"
#define ERR_OBJECT_PROPERTY "Expecting property name enclosed in double quotes"
#define ERR_OBJECT_PROPERTY_FIRST "Expecting property name enclosed in double quotes or '}'"
#define ERR_OBJECT_PROPERTY_DELIMITER "Expecting ':' delimiter"
#define ERR_STRING_UNTERMINATED "Unterminated string starting at"
#define ERR_STRING_CONTROL "Invalid control character %r at"
#define ERR_STRING_ESC1 "Invalid \\X escape sequence %r"
#define ERR_STRING_ESC4 "Invalid \\uXXXX escape sequence"
#define FOR_JSON_METHOD_NAME "for_json"
#define ASDICT_METHOD_NAME "_asdict"


typedef struct _PyScannerObject {
    PyObject_HEAD
    PyObject *encoding;
    PyObject *strict_bool;
    int strict;
    PyObject *object_hook;
    PyObject *pairs_hook;
    PyObject *parse_float;
    PyObject *parse_int;
    PyObject *parse_constant;
    PyObject *memo;
} PyScannerObject;

static PyMemberDef scanner_members[] = {
    {"encoding", T_OBJECT, offsetof(PyScannerObject, encoding), READONLY, "encoding"},
    {"strict", T_OBJECT, offsetof(PyScannerObject, strict_bool), READONLY, "strict"},
    {"object_hook", T_OBJECT, offsetof(PyScannerObject, object_hook), READONLY, "object_hook"},
    {"object_pairs_hook", T_OBJECT, offsetof(PyScannerObject, pairs_hook), READONLY, "object_pairs_hook"},
    {"parse_float", T_OBJECT, offsetof(PyScannerObject, parse_float), READONLY, "parse_float"},
    {"parse_int", T_OBJECT, offsetof(PyScannerObject, parse_int), READONLY, "parse_int"},
    {"parse_constant", T_OBJECT, offsetof(PyScannerObject, parse_constant), READONLY, "parse_constant"},
    {NULL}
};

typedef struct _PyEncoderObject {
    PyObject_HEAD
    PyObject *markers;
    PyObject *defaultfn;
    PyObject *encoder;
    PyObject *indent;
    PyObject *key_separator;
    PyObject *item_separator;
    PyObject *sort_keys;
    PyObject *key_memo;
    PyObject *encoding;
    PyObject *Decimal;
    PyObject *skipkeys_bool;
    int skipkeys;
    int fast_encode;
    /* 0, JSON_ALLOW_NAN, JSON_IGNORE_NAN */
    int allow_or_ignore_nan;
    int use_decimal;
    int namedtuple_as_object;
    int tuple_as_array;
    int iterable_as_array;
    PyObject *max_long_size;
    PyObject *min_long_size;
    PyObject *item_sort_key;
    PyObject *item_sort_kw;
    int for_json;
} PyEncoderObject;

static PyMemberDef encoder_members[] = {
    {"markers", T_OBJECT, offsetof(PyEncoderObject, markers), READONLY, "markers"},
    {"default", T_OBJECT, offsetof(PyEncoderObject, defaultfn), READONLY, "default"},
    {"encoder", T_OBJECT, offsetof(PyEncoderObject, encoder), READONLY, "encoder"},
    {"encoding", T_OBJECT, offsetof(PyEncoderObject, encoder), READONLY, "encoding"},
    {"indent", T_OBJECT, offsetof(PyEncoderObject, indent), READONLY, "indent"},
    {"key_separator", T_OBJECT, offsetof(PyEncoderObject, key_separator), READONLY, "key_separator"},
    {"item_separator", T_OBJECT, offsetof(PyEncoderObject, item_separator), READONLY, "item_separator"},
    {"sort_keys", T_OBJECT, offsetof(PyEncoderObject, sort_keys), READONLY, "sort_keys"},
    /* Python 2.5 does not support T_BOOl */
    {"skipkeys", T_OBJECT, offsetof(PyEncoderObject, skipkeys_bool), READONLY, "skipkeys"},
    {"key_memo", T_OBJECT, offsetof(PyEncoderObject, key_memo), READONLY, "key_memo"},
    {"item_sort_key", T_OBJECT, offsetof(PyEncoderObject, item_sort_key), READONLY, "item_sort_key"},
    {"max_long_size", T_OBJECT, offsetof(PyEncoderObject, max_long_size), READONLY, "max_long_size"},
    {"min_long_size", T_OBJECT, offsetof(PyEncoderObject, min_long_size), READONLY, "min_long_size"},
    {NULL}
};

static PyObject *
join_list_unicode(PyObject *lst);
static PyObject *
JSON_ParseEncoding(PyObject *encoding);
static PyObject *
maybe_quote_bigint(PyEncoderObject* s, PyObject *encoded, PyObject *obj);
static Py_ssize_t
ascii_char_size(JSON_UNICHR c);
static Py_ssize_t
ascii_escape_char(JSON_UNICHR c, char *output, Py_ssize_t chars);
static PyObject *
ascii_escape_unicode(PyObject *pystr);
static PyObject *
ascii_escape_str(PyObject *pystr);
static PyObject *
py_encode_basestring_ascii(PyObject* self UNUSED, PyObject *pystr);
#if PY_MAJOR_VERSION < 3
static PyObject *
join_list_string(PyObject *lst);
static PyObject *
scan_once_str(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr);
static PyObject *
scanstring_str(PyObject *pystr, Py_ssize_t end, char *encoding, int strict, Py_ssize_t *next_end_ptr);
static PyObject *
_parse_object_str(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr);
#endif
static PyObject *
scanstring_unicode(PyObject *pystr, Py_ssize_t end, int strict, Py_ssize_t *next_end_ptr);
static PyObject *
scan_once_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr);
static PyObject *
_build_rval_index_tuple(PyObject *rval, Py_ssize_t idx);
static PyObject *
scanner_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void
scanner_dealloc(PyObject *self);
static int
scanner_clear(PyObject *self);
static PyObject *
encoder_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static void
encoder_dealloc(PyObject *self);
static int
encoder_clear(PyObject *self);
static int
is_raw_json(PyObject *obj);
static PyObject *
encoder_stringify_key(PyEncoderObject *s, PyObject *key);
static int
encoder_listencode_list(PyEncoderObject *s, JSON_Accu *rval, PyObject *seq, Py_ssize_t indent_level);
static int
encoder_listencode_obj(PyEncoderObject *s, JSON_Accu *rval, PyObject *obj, Py_ssize_t indent_level);
static int
encoder_listencode_dict(PyEncoderObject *s, JSON_Accu *rval, PyObject *dct, Py_ssize_t indent_level);
static PyObject *
_encoded_const(PyObject *obj);
static void
raise_errmsg(char *msg, PyObject *s, Py_ssize_t end);
static PyObject *
encoder_encode_string(PyEncoderObject *s, PyObject *obj);
static int
_convertPyInt_AsSsize_t(PyObject *o, Py_ssize_t *size_ptr);
static PyObject *
_convertPyInt_FromSsize_t(Py_ssize_t *size_ptr);
static int
_call_json_method(PyObject *obj, const char *method_name, PyObject **result);
static PyObject *
encoder_encode_float(PyEncoderObject *s, PyObject *obj);
static PyObject *
moduleinit(void);

#define S_CHAR(c) (c >= ' ' && c <= '~' && c != '\\' && c != '"')
#define IS_WHITESPACE(c) (((c) == ' ') || ((c) == '\t') || ((c) == '\n') || ((c) == '\r'))

#define MIN_EXPANSION 6

static PyObject* RawJSONType = NULL;
static int
is_raw_json(PyObject *obj)
{
    return PyObject_IsInstance(obj, RawJSONType) ? 1 : 0;
}

static int
JSON_Accu_Init(JSON_Accu *acc)
{
    /* Lazily allocated */
    acc->large_strings = NULL;
    acc->small_strings = PyList_New(0);
    if (acc->small_strings == NULL)
        return -1;
    return 0;
}

static int
flush_accumulator(JSON_Accu *acc)
{
    Py_ssize_t nsmall = PyList_GET_SIZE(acc->small_strings);
    if (nsmall) {
        int ret;
        PyObject *joined;
        if (acc->large_strings == NULL) {
            acc->large_strings = PyList_New(0);
            if (acc->large_strings == NULL)
                return -1;
        }
#if PY_MAJOR_VERSION >= 3
        joined = join_list_unicode(acc->small_strings);
#else /* PY_MAJOR_VERSION >= 3 */
        joined = join_list_string(acc->small_strings);
#endif /* PY_MAJOR_VERSION < 3 */
        if (joined == NULL)
            return -1;
        if (PyList_SetSlice(acc->small_strings, 0, nsmall, NULL)) {
            Py_DECREF(joined);
            return -1;
        }
        ret = PyList_Append(acc->large_strings, joined);
        Py_DECREF(joined);
        return ret;
    }
    return 0;
}

static int
JSON_Accu_Accumulate(JSON_Accu *acc, PyObject *unicode)
{
    Py_ssize_t nsmall;
#if PY_MAJOR_VERSION >= 3
    assert(PyUnicode_Check(unicode));
#else /* PY_MAJOR_VERSION >= 3 */
    assert(PyString_Check(unicode) || PyUnicode_Check(unicode));
#endif /* PY_MAJOR_VERSION < 3 */

    if (PyList_Append(acc->small_strings, unicode))
        return -1;
    nsmall = PyList_GET_SIZE(acc->small_strings);
    /* Each item in a list of unicode objects has an overhead (in 64-bit
     * builds) of:
     *   - 8 bytes for the list slot
     *   - 56 bytes for the header of the unicode object
     * that is, 64 bytes.  100000 such objects waste more than 6MB
     * compared to a single concatenated string.
     */
    if (nsmall < 100000)
        return 0;
    return flush_accumulator(acc);
}

static PyObject *
JSON_Accu_FinishAsList(JSON_Accu *acc)
{
    int ret;
    PyObject *res;

    ret = flush_accumulator(acc);
    Py_CLEAR(acc->small_strings);
    if (ret) {
        Py_CLEAR(acc->large_strings);
        return NULL;
    }
    res = acc->large_strings;
    acc->large_strings = NULL;
    if (res == NULL)
        return PyList_New(0);
    return res;
}

static void
JSON_Accu_Destroy(JSON_Accu *acc)
{
    Py_CLEAR(acc->small_strings);
    Py_CLEAR(acc->large_strings);
}

static int
IS_DIGIT(JSON_UNICHR c)
{
    return c >= '0' && c <= '9';
}

static PyObject *
maybe_quote_bigint(PyEncoderObject* s, PyObject *encoded, PyObject *obj)
{
    if (s->max_long_size != Py_None && s->min_long_size != Py_None) {
        if (PyObject_RichCompareBool(obj, s->max_long_size, Py_GE) ||
            PyObject_RichCompareBool(obj, s->min_long_size, Py_LE)) {
#if PY_MAJOR_VERSION >= 3
            PyObject* quoted = PyUnicode_FromFormat("\"%U\"", encoded);
#else
            PyObject* quoted = PyString_FromFormat("\"%s\"",
                                                   PyString_AsString(encoded));
#endif
            Py_DECREF(encoded);
            encoded = quoted;
        }
    }

    return encoded;
}

static int
_call_json_method(PyObject *obj, const char *method_name, PyObject **result)
{
    int rval = 0;
    PyObject *method = PyObject_GetAttrString(obj, method_name);
    if (method == NULL) {
        PyErr_Clear();
        return 0;
    }
    if (PyCallable_Check(method)) {
        PyObject *tmp = PyObject_CallNoArgs(method);
        if (tmp == NULL && PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
        } else {
            // This will set result to NULL if a TypeError occurred,
            // which must be checked by the caller
            *result = tmp;
            rval = 1;
        }
    }
    Py_DECREF(method);
    return rval;
}

static int
_convertPyInt_AsSsize_t(PyObject *o, Py_ssize_t *size_ptr)
{
    /* PyObject to Py_ssize_t converter */
    *size_ptr = PyInt_AsSsize_t(o);
    if (*size_ptr == -1 && PyErr_Occurred())
        return 0;
    return 1;
}

static PyObject *
_convertPyInt_FromSsize_t(Py_ssize_t *size_ptr)
{
    /* Py_ssize_t to PyObject converter */
    return PyInt_FromSsize_t(*size_ptr);
}

static Py_ssize_t
ascii_escape_char(JSON_UNICHR c, char *output, Py_ssize_t chars)
{
    /* Escape unicode code point c to ASCII escape sequences
    in char *output. output must have at least 12 bytes unused to
    accommodate an escaped surrogate pair "\uXXXX\uXXXX" */
    if (S_CHAR(c)) {
        output[chars++] = (char)c;
    }
    else {
        output[chars++] = '\\';
        switch (c) {
            case '\\': output[chars++] = (char)c; break;
            case '"': output[chars++] = (char)c; break;
            case '\b': output[chars++] = 'b'; break;
            case '\f': output[chars++] = 'f'; break;
            case '\n': output[chars++] = 'n'; break;
            case '\r': output[chars++] = 'r'; break;
            case '\t': output[chars++] = 't'; break;
            default:
#if PY_MAJOR_VERSION >= 3 || defined(Py_UNICODE_WIDE)
                if (c >= 0x10000) {
                    /* UTF-16 surrogate pair */
                    JSON_UNICHR v = c - 0x10000;
                    c = 0xd800 | ((v >> 10) & 0x3ff);
                    output[chars++] = 'u';
                    output[chars++] = "0123456789abcdef"[(c >> 12) & 0xf];
                    output[chars++] = "0123456789abcdef"[(c >>  8) & 0xf];
                    output[chars++] = "0123456789abcdef"[(c >>  4) & 0xf];
                    output[chars++] = "0123456789abcdef"[(c      ) & 0xf];
                    c = 0xdc00 | (v & 0x3ff);
                    output[chars++] = '\\';
                }
#endif
                output[chars++] = 'u';
                output[chars++] = "0123456789abcdef"[(c >> 12) & 0xf];
                output[chars++] = "0123456789abcdef"[(c >>  8) & 0xf];
                output[chars++] = "0123456789abcdef"[(c >>  4) & 0xf];
                output[chars++] = "0123456789abcdef"[(c      ) & 0xf];
        }
    }
    return chars;
}

static Py_ssize_t
ascii_char_size(JSON_UNICHR c)
{
    if (S_CHAR(c)) {
        return 1;
    }
    else if (c == '\\' ||
               c == '"'  ||
               c == '\b' ||
               c == '\f' ||
               c == '\n' ||
               c == '\r' ||
               c == '\t') {
        return 2;
    }
#if PY_MAJOR_VERSION >= 3 || defined(Py_UNICODE_WIDE)
    else if (c >= 0x10000U) {
        return 2 * MIN_EXPANSION;
    }
#endif
    else {
        return MIN_EXPANSION;
    }
}

static PyObject *
ascii_escape_unicode(PyObject *pystr)
{
    /* Take a PyUnicode pystr and return a new ASCII-only escaped PyString */
    Py_ssize_t i;
    Py_ssize_t input_chars = PyUnicode_GET_LENGTH(pystr);
    Py_ssize_t output_size = 2;
    Py_ssize_t chars;
    PY2_UNUSED int kind = PyUnicode_KIND(pystr);
    void *data = PyUnicode_DATA(pystr);
    PyObject *rval;
    char *output;

    output_size = 2;
    for (i = 0; i < input_chars; i++) {
        output_size += ascii_char_size(PyUnicode_READ(kind, data, i));
    }
#if PY_MAJOR_VERSION >= 3
    rval = PyUnicode_New(output_size, 127);
    if (rval == NULL) {
        return NULL;
    }
    assert(PyUnicode_KIND(rval) == PyUnicode_1BYTE_KIND);
    output = (char *)PyUnicode_DATA(rval);
#else
    rval = PyString_FromStringAndSize(NULL, output_size);
    if (rval == NULL) {
        return NULL;
    }
    output = PyString_AS_STRING(rval);
#endif
    chars = 0;
    output[chars++] = '"';
    for (i = 0; i < input_chars; i++) {
        chars = ascii_escape_char(PyUnicode_READ(kind, data, i), output, chars);
    }
    output[chars++] = '"';
    assert(chars == output_size);
    return rval;
}

#if PY_MAJOR_VERSION >= 3

static PyObject *
ascii_escape_str(PyObject *pystr)
{
    PyObject *rval;
    PyObject *input = PyUnicode_DecodeUTF8(PyBytes_AS_STRING(pystr), PyBytes_GET_SIZE(pystr), NULL);
    if (input == NULL)
        return NULL;
    rval = ascii_escape_unicode(input);
    Py_DECREF(input);
    return rval;
}

#else /* PY_MAJOR_VERSION >= 3 */

static PyObject *
ascii_escape_str(PyObject *pystr)
{
    /* Take a PyString pystr and return a new ASCII-only escaped PyString */
    Py_ssize_t i;
    Py_ssize_t input_chars;
    Py_ssize_t output_size;
    Py_ssize_t chars;
    PyObject *rval;
    char *output;
    char *input_str;

    input_chars = PyString_GET_SIZE(pystr);
    input_str = PyString_AS_STRING(pystr);
    output_size = 2;

    /* Fast path for a string that's already ASCII */
    for (i = 0; i < input_chars; i++) {
        JSON_UNICHR c = (JSON_UNICHR)input_str[i];
        if (c > 0x7f) {
            /* We hit a non-ASCII character, bail to unicode mode */
            PyObject *uni;
            uni = PyUnicode_DecodeUTF8(input_str, input_chars, "strict");
            if (uni == NULL) {
                return NULL;
            }
            rval = ascii_escape_unicode(uni);
            Py_DECREF(uni);
            return rval;
        }
        output_size += ascii_char_size(c);
    }

    rval = PyString_FromStringAndSize(NULL, output_size);
    if (rval == NULL) {
        return NULL;
    }
    chars = 0;
    output = PyString_AS_STRING(rval);
    output[chars++] = '"';
    for (i = 0; i < input_chars; i++) {
        chars = ascii_escape_char((JSON_UNICHR)input_str[i], output, chars);
    }
    output[chars++] = '"';
    assert(chars == output_size);
    return rval;
}
#endif /* PY_MAJOR_VERSION < 3 */

static PyObject *
encoder_stringify_key(PyEncoderObject *s, PyObject *key)
{
    if (PyUnicode_Check(key)) {
        Py_INCREF(key);
        return key;
    }
#if PY_MAJOR_VERSION >= 3
    else if (PyBytes_Check(key) && s->encoding != NULL) {
        const char *encoding = PyUnicode_AsUTF8(s->encoding);
        if (encoding == NULL)
            return NULL;
        return PyUnicode_Decode(
            PyBytes_AS_STRING(key),
            PyBytes_GET_SIZE(key),
            encoding,
            NULL);
    }
#else /* PY_MAJOR_VERSION >= 3 */
    else if (PyString_Check(key)) {
        Py_INCREF(key);
        return key;
    }
#endif /* PY_MAJOR_VERSION < 3 */
    else if (PyFloat_Check(key)) {
        return encoder_encode_float(s, key);
    }
    else if (key == Py_True || key == Py_False || key == Py_None) {
        /* This must come before the PyInt_Check because
           True and False are also 1 and 0.*/
        return _encoded_const(key);
    }
    else if (PyInt_Check(key) || PyLong_Check(key)) {
        if (!(PyInt_CheckExact(key) || PyLong_CheckExact(key))) {
            /* See #118, do not trust custom str/repr */
            PyObject *res;
            PyObject *tmp = PyObject_CallOneArg((PyObject *)&PyLong_Type, key);
            if (tmp == NULL) {
                return NULL;
            }
            res = PyObject_Str(tmp);
            Py_DECREF(tmp);
            return res;
        }
        else {
            return PyObject_Str(key);
        }
    }
    else if (s->use_decimal && PyObject_TypeCheck(key, (PyTypeObject *)s->Decimal)) {
        return PyObject_Str(key);
    }
    if (s->skipkeys) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    PyErr_Format(PyExc_TypeError,
                 "keys must be str, int, float, bool or None, "
                 "not %.100s", key->ob_type->tp_name);
    return NULL;
}

static PyObject *
encoder_dict_iteritems(PyEncoderObject *s, PyObject *dct)
{
    PyObject *items;
    PyObject *iter = NULL;
    PyObject *lst = NULL;
    PyObject *item = NULL;
    PyObject *kstr = NULL;
    PyObject *sortfun = NULL;
    PyObject *sortres;
    static PyObject *sortargs = NULL;

    if (sortargs == NULL) {
        sortargs = PyTuple_New(0);
        if (sortargs == NULL)
            return NULL;
    }

    if (PyDict_CheckExact(dct))
        items = PyDict_Items(dct);
    else
        items = PyMapping_Items(dct);
    if (items == NULL)
        return NULL;
    iter = PyObject_GetIter(items);
    Py_DECREF(items);
    if (iter == NULL)
        return NULL;
    if (s->item_sort_kw == Py_None)
        return iter;
    lst = PyList_New(0);
    if (lst == NULL)
        goto bail;
    while ((item = PyIter_Next(iter))) {
        PyObject *key, *value;
        if (!PyTuple_Check(item) || Py_SIZE(item) != 2) {
            PyErr_SetString(PyExc_ValueError, "items must return 2-tuples");
            goto bail;
        }
        key = PyTuple_GET_ITEM(item, 0);
        if (key == NULL)
            goto bail;
#if PY_MAJOR_VERSION < 3
        else if (PyString_Check(key)) {
            /* item can be added as-is */
        }
#endif /* PY_MAJOR_VERSION < 3 */
        else if (PyUnicode_Check(key)) {
            /* item can be added as-is */
        }
        else {
            PyObject *tpl;
            kstr = encoder_stringify_key(s, key);
            if (kstr == NULL)
                goto bail;
            else if (kstr == Py_None) {
                /* skipkeys */
                Py_DECREF(kstr);
                continue;
            }
            value = PyTuple_GET_ITEM(item, 1);
            if (value == NULL)
                goto bail;
            tpl = PyTuple_Pack(2, kstr, value);
            if (tpl == NULL)
                goto bail;
            Py_CLEAR(kstr);
            Py_DECREF(item);
            item = tpl;
        }
        if (PyList_Append(lst, item))
            goto bail;
        Py_DECREF(item);
    }
    Py_CLEAR(iter);
    if (PyErr_Occurred())
        goto bail;
    sortfun = PyObject_GetAttrString(lst, "sort");
    if (sortfun == NULL)
        goto bail;
    sortres = PyObject_Call(sortfun, sortargs, s->item_sort_kw);
    if (!sortres)
        goto bail;
    Py_DECREF(sortres);
    Py_CLEAR(sortfun);
    iter = PyObject_GetIter(lst);
    Py_CLEAR(lst);
    return iter;
bail:
    Py_XDECREF(sortfun);
    Py_XDECREF(kstr);
    Py_XDECREF(item);
    Py_XDECREF(lst);
    Py_XDECREF(iter);
    return NULL;
}

/* Use JSONDecodeError exception to raise a nice looking ValueError subclass */
static PyObject *JSONDecodeError = NULL;
static void
raise_errmsg(char *msg, PyObject *s, Py_ssize_t end)
{
    PyObject *exc = PyObject_CallFunction(JSONDecodeError, "(zOO&)", msg, s, _convertPyInt_FromSsize_t, &end);
    if (exc) {
        PyErr_SetObject(JSONDecodeError, exc);
        Py_DECREF(exc);
    }
}

static PyObject *
join_list_unicode(PyObject *lst)
{
    /* return u''.join(lst) */
    return PyUnicode_Join(JSON_EmptyUnicode, lst);
}

#if PY_MAJOR_VERSION >= 3
#define join_list_string join_list_unicode
#else /* PY_MAJOR_VERSION >= 3 */
static PyObject *
join_list_string(PyObject *lst)
{
    /* return ''.join(lst) */
    static PyObject *joinfn = NULL;
    if (joinfn == NULL) {
        joinfn = PyObject_GetAttrString(JSON_EmptyStr, "join");
        if (joinfn == NULL)
            return NULL;
    }
    return PyObject_CallOneArg(joinfn, lst);
}
#endif /* PY_MAJOR_VERSION < 3 */

static PyObject *
_build_rval_index_tuple(PyObject *rval, Py_ssize_t idx)
{
    /* return (rval, idx) tuple, stealing reference to rval */
    PyObject *tpl;
    PyObject *pyidx;
    /*
    steal a reference to rval, returns (rval, idx)
    */
    if (rval == NULL) {
        assert(PyErr_Occurred());
        return NULL;
    }
    pyidx = PyInt_FromSsize_t(idx);
    if (pyidx == NULL) {
        Py_DECREF(rval);
        return NULL;
    }
    tpl = PyTuple_New(2);
    if (tpl == NULL) {
        Py_DECREF(pyidx);
        Py_DECREF(rval);
        return NULL;
    }
    PyTuple_SET_ITEM(tpl, 0, rval);
    PyTuple_SET_ITEM(tpl, 1, pyidx);
    return tpl;
}

#define APPEND_OLD_CHUNK \
    if (chunk != NULL) { \
        if (chunks == NULL) { \
            chunks = PyList_New(0); \
            if (chunks == NULL) { \
                goto bail; \
            } \
        } \
        if (PyList_Append(chunks, chunk)) { \
            goto bail; \
        } \
        Py_CLEAR(chunk); \
    }

#if PY_MAJOR_VERSION < 3
static PyObject *
scanstring_str(PyObject *pystr, Py_ssize_t end, char *encoding, int strict, Py_ssize_t *next_end_ptr)
{
    /* Read the JSON string from PyString pystr.
    end is the index of the first character after the quote.
    encoding is the encoding of pystr (must be an ASCII superset)
    if strict is zero then literal control characters are allowed
    *next_end_ptr is a return-by-reference index of the character
        after the end quote

    Return value is a new PyString (if ASCII-only) or PyUnicode
    */
    PyObject *rval;
    Py_ssize_t len = PyString_GET_SIZE(pystr);
    Py_ssize_t begin = end - 1;
    Py_ssize_t next = begin;
    int has_unicode = 0;
    char *buf = PyString_AS_STRING(pystr);
    PyObject *chunks = NULL;
    PyObject *chunk = NULL;
    PyObject *strchunk = NULL;

    if (len == end) {
        raise_errmsg(ERR_STRING_UNTERMINATED, pystr, begin);
        goto bail;
    }
    else if (end < 0 || len < end) {
        PyErr_SetString(PyExc_ValueError, "end is out of bounds");
        goto bail;
    }
    while (1) {
        /* Find the end of the string or the next escape */
        Py_UNICODE c = 0;
        for (next = end; next < len; next++) {
            c = (unsigned char)buf[next];
            if (c == '"' || c == '\\') {
                break;
            }
            else if (strict && c <= 0x1f) {
                raise_errmsg(ERR_STRING_CONTROL, pystr, next);
                goto bail;
            }
            else if (c > 0x7f) {
                has_unicode = 1;
            }
        }
        if (!(c == '"' || c == '\\')) {
            raise_errmsg(ERR_STRING_UNTERMINATED, pystr, begin);
            goto bail;
        }
        /* Pick up this chunk if it's not zero length */
        if (next != end) {
            APPEND_OLD_CHUNK
            strchunk = PyString_FromStringAndSize(&buf[end], next - end);
            if (strchunk == NULL) {
                goto bail;
            }
            if (has_unicode) {
                chunk = PyUnicode_FromEncodedObject(strchunk, encoding, NULL);
                Py_DECREF(strchunk);
                if (chunk == NULL) {
                    goto bail;
                }
            }
            else {
                chunk = strchunk;
            }
        }
        next++;
        if (c == '"') {
            end = next;
            break;
        }
        if (next == len) {
            raise_errmsg(ERR_STRING_UNTERMINATED, pystr, begin);
            goto bail;
        }
        c = buf[next];
        if (c != 'u') {
            /* Non-unicode backslash escapes */
            end = next + 1;
            switch (c) {
                case '"': break;
                case '\\': break;
                case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default: c = 0;
            }
            if (c == 0) {
                raise_errmsg(ERR_STRING_ESC1, pystr, end - 2);
                goto bail;
            }
        }
        else {
            c = 0;
            next++;
            end = next + 4;
            if (end >= len) {
                raise_errmsg(ERR_STRING_ESC4, pystr, next - 1);
                goto bail;
            }
            /* Decode 4 hex digits */
            for (; next < end; next++) {
                JSON_UNICHR digit = (JSON_UNICHR)buf[next];
                c <<= 4;
                switch (digit) {
                    case '0': case '1': case '2': case '3': case '4':
                    case '5': case '6': case '7': case '8': case '9':
                        c |= (digit - '0'); break;
                    case 'a': case 'b': case 'c': case 'd': case 'e':
                    case 'f':
                        c |= (digit - 'a' + 10); break;
                    case 'A': case 'B': case 'C': case 'D': case 'E':
                    case 'F':
                        c |= (digit - 'A' + 10); break;
                    default:
                        raise_errmsg(ERR_STRING_ESC4, pystr, end - 5);
                        goto bail;
                }
            }
#if defined(Py_UNICODE_WIDE)
            /* Surrogate pair */
            if ((c & 0xfc00) == 0xd800) {
                if (end + 6 < len && buf[next] == '\\' && buf[next+1] == 'u') {
                    JSON_UNICHR c2 = 0;
                    end += 6;
                    /* Decode 4 hex digits */
                    for (next += 2; next < end; next++) {
                        c2 <<= 4;
                        JSON_UNICHR digit = buf[next];
                        switch (digit) {
                        case '0': case '1': case '2': case '3': case '4':
                        case '5': case '6': case '7': case '8': case '9':
                            c2 |= (digit - '0'); break;
                        case 'a': case 'b': case 'c': case 'd': case 'e':
                        case 'f':
                            c2 |= (digit - 'a' + 10); break;
                        case 'A': case 'B': case 'C': case 'D': case 'E':
                        case 'F':
                            c2 |= (digit - 'A' + 10); break;
                        default:
                            raise_errmsg(ERR_STRING_ESC4, pystr, end - 5);
                            goto bail;
                        }
                    }
                    if ((c2 & 0xfc00) != 0xdc00) {
                        /* not a low surrogate, rewind */
                        end -= 6;
                        next = end;
                    }
                    else {
                        c = 0x10000 + (((c - 0xd800) << 10) | (c2 - 0xdc00));
                    }
                }
            }
#endif /* Py_UNICODE_WIDE */
        }
        if (c > 0x7f) {
            has_unicode = 1;
        }
        APPEND_OLD_CHUNK
        if (has_unicode) {
            chunk = PyUnicode_FromOrdinal(c);
            if (chunk == NULL) {
                goto bail;
            }
        }
        else {
            char c_char = Py_CHARMASK(c);
            chunk = PyString_FromStringAndSize(&c_char, 1);
            if (chunk == NULL) {
                goto bail;
            }
        }
    }

    if (chunks == NULL) {
        if (chunk != NULL)
            rval = chunk;
        else {
            rval = JSON_EmptyStr;
            Py_INCREF(rval);
        }
    }
    else {
        APPEND_OLD_CHUNK
        rval = join_list_string(chunks);
        if (rval == NULL) {
            goto bail;
        }
        Py_CLEAR(chunks);
    }

    *next_end_ptr = end;
    return rval;
bail:
    *next_end_ptr = -1;
    Py_XDECREF(chunk);
    Py_XDECREF(chunks);
    return NULL;
}
#endif /* PY_MAJOR_VERSION < 3 */

static PyObject *
scanstring_unicode(PyObject *pystr, Py_ssize_t end, int strict, Py_ssize_t *next_end_ptr)
{
    /* Read the JSON string from PyUnicode pystr.
    end is the index of the first character after the quote.
    if strict is zero then literal control characters are allowed
    *next_end_ptr is a return-by-reference index of the character
        after the end quote

    Return value is a new PyUnicode
    */
    PyObject *rval;
    Py_ssize_t begin = end - 1;
    Py_ssize_t next = begin;
    PY2_UNUSED int kind = PyUnicode_KIND(pystr);
    Py_ssize_t len = PyUnicode_GET_LENGTH(pystr);
    void *buf = PyUnicode_DATA(pystr);
    PyObject *chunks = NULL;
    PyObject *chunk = NULL;

    if (len == end) {
        raise_errmsg(ERR_STRING_UNTERMINATED, pystr, begin);
        goto bail;
    }
    else if (end < 0 || len < end) {
        PyErr_SetString(PyExc_ValueError, "end is out of bounds");
        goto bail;
    }
    while (1) {
        /* Find the end of the string or the next escape */
        JSON_UNICHR c = 0;
        for (next = end; next < len; next++) {
            c = PyUnicode_READ(kind, buf, next);
            if (c == '"' || c == '\\') {
                break;
            }
            else if (strict && c <= 0x1f) {
                raise_errmsg(ERR_STRING_CONTROL, pystr, next);
                goto bail;
            }
        }
        if (!(c == '"' || c == '\\')) {
            raise_errmsg(ERR_STRING_UNTERMINATED, pystr, begin);
            goto bail;
        }
        /* Pick up this chunk if it's not zero length */
        if (next != end) {
            APPEND_OLD_CHUNK
#if PY_MAJOR_VERSION < 3
            chunk = PyUnicode_FromUnicode(&((const Py_UNICODE *)buf)[end], next - end);
#else
            chunk = PyUnicode_Substring(pystr, end, next);
#endif
            if (chunk == NULL) {
                goto bail;
            }
        }
        next++;
        if (c == '"') {
            end = next;
            break;
        }
        if (next == len) {
            raise_errmsg(ERR_STRING_UNTERMINATED, pystr, begin);
            goto bail;
        }
        c = PyUnicode_READ(kind, buf, next);
        if (c != 'u') {
            /* Non-unicode backslash escapes */
            end = next + 1;
            switch (c) {
                case '"': break;
                case '\\': break;
                case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default: c = 0;
            }
            if (c == 0) {
                raise_errmsg(ERR_STRING_ESC1, pystr, end - 2);
                goto bail;
            }
        }
        else {
            c = 0;
            next++;
            end = next + 4;
            if (end >= len) {
                raise_errmsg(ERR_STRING_ESC4, pystr, next - 1);
                goto bail;
            }
            /* Decode 4 hex digits */
            for (; next < end; next++) {
                JSON_UNICHR digit = PyUnicode_READ(kind, buf, next);
                c <<= 4;
                switch (digit) {
                    case '0': case '1': case '2': case '3': case '4':
                    case '5': case '6': case '7': case '8': case '9':
                        c |= (digit - '0'); break;
                    case 'a': case 'b': case 'c': case 'd': case 'e':
                    case 'f':
                        c |= (digit - 'a' + 10); break;
                    case 'A': case 'B': case 'C': case 'D': case 'E':
                    case 'F':
                        c |= (digit - 'A' + 10); break;
                    default:
                        raise_errmsg(ERR_STRING_ESC4, pystr, end - 5);
                        goto bail;
                }
            }
#if PY_MAJOR_VERSION >= 3 || defined(Py_UNICODE_WIDE)
            /* Surrogate pair */
            if ((c & 0xfc00) == 0xd800) {
                JSON_UNICHR c2 = 0;
                if (end + 6 < len &&
                    PyUnicode_READ(kind, buf, next) == '\\' &&
                    PyUnicode_READ(kind, buf, next + 1) == 'u') {
                    end += 6;
                    /* Decode 4 hex digits */
                    for (next += 2; next < end; next++) {
                        JSON_UNICHR digit = PyUnicode_READ(kind, buf, next);
                        c2 <<= 4;
                        switch (digit) {
                        case '0': case '1': case '2': case '3': case '4':
                        case '5': case '6': case '7': case '8': case '9':
                            c2 |= (digit - '0'); break;
                        case 'a': case 'b': case 'c': case 'd': case 'e':
                        case 'f':
                            c2 |= (digit - 'a' + 10); break;
                        case 'A': case 'B': case 'C': case 'D': case 'E':
                        case 'F':
                            c2 |= (digit - 'A' + 10); break;
                        default:
                            raise_errmsg(ERR_STRING_ESC4, pystr, end - 5);
                            goto bail;
                        }
                    }
                    if ((c2 & 0xfc00) != 0xdc00) {
                        /* not a low surrogate, rewind */
                        end -= 6;
                        next = end;
                    }
                    else {
                        c = 0x10000 + (((c - 0xd800) << 10) | (c2 - 0xdc00));
                    }
                }
            }
#endif
        }
        APPEND_OLD_CHUNK
        chunk = PyUnicode_FromOrdinal(c);
        if (chunk == NULL) {
            goto bail;
        }
    }

    if (chunks == NULL) {
        if (chunk != NULL)
            rval = chunk;
        else {
            rval = JSON_EmptyUnicode;
            Py_INCREF(rval);
        }
    }
    else {
        APPEND_OLD_CHUNK
        rval = join_list_unicode(chunks);
        if (rval == NULL) {
            goto bail;
        }
        Py_CLEAR(chunks);
    }
    *next_end_ptr = end;
    return rval;
bail:
    *next_end_ptr = -1;
    Py_XDECREF(chunk);
    Py_XDECREF(chunks);
    return NULL;
}

PyDoc_STRVAR(pydoc_scanstring,
    "scanstring(basestring, end, encoding, strict=True) -> (str, end)\n"
    "\n"
    "Scan the string s for a JSON string. End is the index of the\n"
    "character in s after the quote that started the JSON string.\n"
    "Unescapes all valid JSON string escape sequences and raises ValueError\n"
    "on attempt to decode an invalid string. If strict is False then literal\n"
    "control characters are allowed in the string.\n"
    "\n"
    "Returns a tuple of the decoded string and the index of the character in s\n"
    "after the end quote."
);

static PyObject *
py_scanstring(PyObject* self UNUSED, PyObject *args)
{
    PyObject *pystr;
    PyObject *rval;
    Py_ssize_t end;
    Py_ssize_t next_end = -1;
    char *encoding = NULL;
    int strict = 1;
    if (!PyArg_ParseTuple(args, "OO&|zi:scanstring", &pystr, _convertPyInt_AsSsize_t, &end, &encoding, &strict)) {
        return NULL;
    }
    if (encoding == NULL) {
        encoding = DEFAULT_ENCODING;
    }
    if (PyUnicode_Check(pystr)) {
        if (PyUnicode_READY(pystr))
            return NULL;
        rval = scanstring_unicode(pystr, end, strict, &next_end);
    }
#if PY_MAJOR_VERSION < 3
    /* Using a bytes input is unsupported for scanning in Python 3.
       It is coerced to str in the decoder before it gets here. */
    else if (PyString_Check(pystr)) {
        rval = scanstring_str(pystr, end, encoding, strict, &next_end);
    }
#endif
    else {
        PyErr_Format(PyExc_TypeError,
                     "first argument must be a string, not %.80s",
                     Py_TYPE(pystr)->tp_name);
        return NULL;
    }
    return _build_rval_index_tuple(rval, next_end);
}

PyDoc_STRVAR(pydoc_encode_basestring_ascii,
    "encode_basestring_ascii(basestring) -> str\n"
    "\n"
    "Return an ASCII-only JSON representation of a Python string"
);

static PyObject *
py_encode_basestring_ascii(PyObject* self UNUSED, PyObject *pystr)
{
    /* Return an ASCII-only JSON representation of a Python string */
    /* METH_O */
    if (PyBytes_Check(pystr)) {
        return ascii_escape_str(pystr);
    }
    else if (PyUnicode_Check(pystr)) {
        if (PyUnicode_READY(pystr))
            return NULL;
        return ascii_escape_unicode(pystr);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "first argument must be a string, not %.80s",
                     Py_TYPE(pystr)->tp_name);
        return NULL;
    }
}

static void
scanner_dealloc(PyObject *self)
{
    /* bpo-31095: UnTrack is needed before calling any callbacks */
    PyObject_GC_UnTrack(self);
    scanner_clear(self);
    Py_TYPE(self)->tp_free(self);
}

static int
scanner_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyScannerObject *s;
    assert(PyScanner_Check(self));
    s = (PyScannerObject *)self;
    Py_VISIT(s->encoding);
    Py_VISIT(s->strict_bool);
    Py_VISIT(s->object_hook);
    Py_VISIT(s->pairs_hook);
    Py_VISIT(s->parse_float);
    Py_VISIT(s->parse_int);
    Py_VISIT(s->parse_constant);
    Py_VISIT(s->memo);
    return 0;
}

static int
scanner_clear(PyObject *self)
{
    PyScannerObject *s;
    assert(PyScanner_Check(self));
    s = (PyScannerObject *)self;
    Py_CLEAR(s->encoding);
    Py_CLEAR(s->strict_bool);
    Py_CLEAR(s->object_hook);
    Py_CLEAR(s->pairs_hook);
    Py_CLEAR(s->parse_float);
    Py_CLEAR(s->parse_int);
    Py_CLEAR(s->parse_constant);
    Py_CLEAR(s->memo);
    return 0;
}

#if PY_MAJOR_VERSION < 3
static PyObject *
_parse_object_str(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read a JSON object from PyString pystr.
    idx is the index of the first character after the opening curly brace.
    *next_idx_ptr is a return-by-reference index to the first character after
        the closing curly brace.

    Returns a new PyObject (usually a dict, but object_hook or
    object_pairs_hook can change that)
    */
    char *str = PyString_AS_STRING(pystr);
    Py_ssize_t end_idx = PyString_GET_SIZE(pystr) - 1;
    PyObject *rval = NULL;
    PyObject *pairs = NULL;
    PyObject *item;
    PyObject *key = NULL;
    PyObject *val = NULL;
    char *encoding = PyString_AS_STRING(s->encoding);
    int has_pairs_hook = (s->pairs_hook != Py_None);
    int did_parse = 0;
    Py_ssize_t next_idx;
    if (has_pairs_hook) {
        pairs = PyList_New(0);
        if (pairs == NULL)
            return NULL;
    }
    else {
        rval = PyDict_New();
        if (rval == NULL)
            return NULL;
    }

    /* skip whitespace after { */
    while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

    /* only loop if the object is non-empty */
    if (idx <= end_idx && str[idx] != '}') {
        int trailing_delimiter = 0;
        while (idx <= end_idx) {
            PyObject *memokey;
            trailing_delimiter = 0;

            /* read key */
            if (str[idx] != '"') {
                raise_errmsg(ERR_OBJECT_PROPERTY, pystr, idx);
                goto bail;
            }
            key = scanstring_str(pystr, idx + 1, encoding, s->strict, &next_idx);
            if (key == NULL)
                goto bail;
            memokey = PyDict_GetItem(s->memo, key);
            if (memokey != NULL) {
                Py_INCREF(memokey);
                Py_DECREF(key);
                key = memokey;
            }
            else {
                if (PyDict_SetItem(s->memo, key, key) < 0)
                    goto bail;
            }
            idx = next_idx;

            /* skip whitespace between key and : delimiter, read :, skip whitespace */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;
            if (idx > end_idx || str[idx] != ':') {
                raise_errmsg(ERR_OBJECT_PROPERTY_DELIMITER, pystr, idx);
                goto bail;
            }
            idx++;
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

            /* read any JSON data type */
            val = scan_once_str(s, pystr, idx, &next_idx);
            if (val == NULL)
                goto bail;

            if (has_pairs_hook) {
                item = PyTuple_Pack(2, key, val);
                if (item == NULL)
                    goto bail;
                Py_CLEAR(key);
                Py_CLEAR(val);
                if (PyList_Append(pairs, item) == -1) {
                    Py_DECREF(item);
                    goto bail;
                }
                Py_DECREF(item);
            }
            else {
                if (PyDict_SetItem(rval, key, val) < 0)
                    goto bail;
                Py_CLEAR(key);
                Py_CLEAR(val);
            }
            idx = next_idx;

            /* skip whitespace before } or , */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

            /* bail if the object is closed or we didn't get the , delimiter */
            did_parse = 1;
            if (idx > end_idx) break;
            if (str[idx] == '}') {
                break;
            }
            else if (str[idx] != ',') {
                raise_errmsg(ERR_OBJECT_DELIMITER, pystr, idx);
                goto bail;
            }
            idx++;

            /* skip whitespace after , delimiter */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;
            trailing_delimiter = 1;
        }
        if (trailing_delimiter) {
            raise_errmsg(ERR_OBJECT_PROPERTY, pystr, idx);
            goto bail;
        }
    }
    /* verify that idx < end_idx, str[idx] should be '}' */
    if (idx > end_idx || str[idx] != '}') {
        if (did_parse) {
            raise_errmsg(ERR_OBJECT_DELIMITER, pystr, idx);
        } else {
            raise_errmsg(ERR_OBJECT_PROPERTY_FIRST, pystr, idx);
        }
        goto bail;
    }

    /* if pairs_hook is not None: rval = object_pairs_hook(pairs) */
    if (s->pairs_hook != Py_None) {
        val = PyObject_CallOneArg(s->pairs_hook, pairs);
        if (val == NULL)
            goto bail;
        Py_DECREF(pairs);
        *next_idx_ptr = idx + 1;
        return val;
    }

    /* if object_hook is not None: rval = object_hook(rval) */
    if (s->object_hook != Py_None) {
        val = PyObject_CallOneArg(s->object_hook, rval);
        if (val == NULL)
            goto bail;
        Py_DECREF(rval);
        rval = val;
        val = NULL;
    }
    *next_idx_ptr = idx + 1;
    return rval;
bail:
    Py_XDECREF(rval);
    Py_XDECREF(key);
    Py_XDECREF(val);
    Py_XDECREF(pairs);
    return NULL;
}
#endif /* PY_MAJOR_VERSION < 3 */

static PyObject *
_parse_object_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read a JSON object from PyUnicode pystr.
    idx is the index of the first character after the opening curly brace.
    *next_idx_ptr is a return-by-reference index to the first character after
        the closing curly brace.

    Returns a new PyObject (usually a dict, but object_hook can change that)
    */
    void *str = PyUnicode_DATA(pystr);
    Py_ssize_t end_idx = PyUnicode_GET_LENGTH(pystr) - 1;
    PY2_UNUSED int kind = PyUnicode_KIND(pystr);
    PyObject *rval = NULL;
    PyObject *pairs = NULL;
    PyObject *item;
    PyObject *key = NULL;
    PyObject *val = NULL;
    int has_pairs_hook = (s->pairs_hook != Py_None);
    int did_parse = 0;
    Py_ssize_t next_idx;

    if (has_pairs_hook) {
        pairs = PyList_New(0);
        if (pairs == NULL)
            return NULL;
    }
    else {
        rval = PyDict_New();
        if (rval == NULL)
            return NULL;
    }

    /* skip whitespace after { */
    while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;

    /* only loop if the object is non-empty */
    if (idx <= end_idx && PyUnicode_READ(kind, str, idx) != '}') {
        int trailing_delimiter = 0;
        while (idx <= end_idx) {
            PyObject *memokey;
            trailing_delimiter = 0;

            /* read key */
            if (PyUnicode_READ(kind, str, idx) != '"') {
                raise_errmsg(ERR_OBJECT_PROPERTY, pystr, idx);
                goto bail;
            }
            key = scanstring_unicode(pystr, idx + 1, s->strict, &next_idx);
            if (key == NULL)
                goto bail;
            memokey = PyDict_GetItem(s->memo, key);
            if (memokey != NULL) {
                Py_INCREF(memokey);
                Py_DECREF(key);
                key = memokey;
            }
            else {
                if (PyDict_SetItem(s->memo, key, key) < 0)
                    goto bail;
            }
            idx = next_idx;

            /* skip whitespace between key and : delimiter, read :, skip
               whitespace */
            while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;
            if (idx > end_idx || PyUnicode_READ(kind, str, idx) != ':') {
                raise_errmsg(ERR_OBJECT_PROPERTY_DELIMITER, pystr, idx);
                goto bail;
            }
            idx++;
            while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;

            /* read any JSON term */
            val = scan_once_unicode(s, pystr, idx, &next_idx);
            if (val == NULL)
                goto bail;

            if (has_pairs_hook) {
                item = PyTuple_Pack(2, key, val);
                if (item == NULL)
                    goto bail;
                Py_CLEAR(key);
                Py_CLEAR(val);
                if (PyList_Append(pairs, item) == -1) {
                    Py_DECREF(item);
                    goto bail;
                }
                Py_DECREF(item);
            }
            else {
                if (PyDict_SetItem(rval, key, val) < 0)
                    goto bail;
                Py_CLEAR(key);
                Py_CLEAR(val);
            }
            idx = next_idx;

            /* skip whitespace before } or , */
            while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;

            /* bail if the object is closed or we didn't get the ,
               delimiter */
            did_parse = 1;
            if (idx > end_idx) break;
            if (PyUnicode_READ(kind, str, idx) == '}') {
                break;
            }
            else if (PyUnicode_READ(kind, str, idx) != ',') {
                raise_errmsg(ERR_OBJECT_DELIMITER, pystr, idx);
                goto bail;
            }
            idx++;

            /* skip whitespace after , delimiter */
            while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;
            trailing_delimiter = 1;
        }
        if (trailing_delimiter) {
            raise_errmsg(ERR_OBJECT_PROPERTY, pystr, idx);
            goto bail;
        }
    }

    /* verify that idx < end_idx, str[idx] should be '}' */
    if (idx > end_idx || PyUnicode_READ(kind, str, idx) != '}') {
        if (did_parse) {
            raise_errmsg(ERR_OBJECT_DELIMITER, pystr, idx);
        } else {
            raise_errmsg(ERR_OBJECT_PROPERTY_FIRST, pystr, idx);
        }
        goto bail;
    }

    /* if pairs_hook is not None: rval = object_pairs_hook(pairs) */
    if (s->pairs_hook != Py_None) {
        val = PyObject_CallOneArg(s->pairs_hook, pairs);
        if (val == NULL)
            goto bail;
        Py_DECREF(pairs);
        *next_idx_ptr = idx + 1;
        return val;
    }

    /* if object_hook is not None: rval = object_hook(rval) */
    if (s->object_hook != Py_None) {
        val = PyObject_CallOneArg(s->object_hook, rval);
        if (val == NULL)
            goto bail;
        Py_DECREF(rval);
        rval = val;
        val = NULL;
    }
    *next_idx_ptr = idx + 1;
    return rval;
bail:
    Py_XDECREF(rval);
    Py_XDECREF(key);
    Py_XDECREF(val);
    Py_XDECREF(pairs);
    return NULL;
}

#if PY_MAJOR_VERSION < 3
static PyObject *
_parse_array_str(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read a JSON array from PyString pystr.
    idx is the index of the first character after the opening brace.
    *next_idx_ptr is a return-by-reference index to the first character after
        the closing brace.

    Returns a new PyList
    */
    char *str = PyString_AS_STRING(pystr);
    Py_ssize_t end_idx = PyString_GET_SIZE(pystr) - 1;
    PyObject *val = NULL;
    PyObject *rval = PyList_New(0);
    Py_ssize_t next_idx;
    if (rval == NULL)
        return NULL;

    /* skip whitespace after [ */
    while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

    /* only loop if the array is non-empty */
    if (idx <= end_idx && str[idx] != ']') {
        int trailing_delimiter = 0;
        while (idx <= end_idx) {
            trailing_delimiter = 0;
            /* read any JSON term and de-tuplefy the (rval, idx) */
            val = scan_once_str(s, pystr, idx, &next_idx);
            if (val == NULL) {
                goto bail;
            }

            if (PyList_Append(rval, val) == -1)
                goto bail;

            Py_CLEAR(val);
            idx = next_idx;

            /* skip whitespace between term and , */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

            /* bail if the array is closed or we didn't get the , delimiter */
            if (idx > end_idx) break;
            if (str[idx] == ']') {
                break;
            }
            else if (str[idx] != ',') {
                raise_errmsg(ERR_ARRAY_DELIMITER, pystr, idx);
                goto bail;
            }
            idx++;

            /* skip whitespace after , */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;
            trailing_delimiter = 1;
        }
        if (trailing_delimiter) {
            raise_errmsg(ERR_EXPECTING_VALUE, pystr, idx);
            goto bail;
        }
    }

    /* verify that idx < end_idx, str[idx] should be ']' */
    if (idx > end_idx || str[idx] != ']') {
        if (PyList_GET_SIZE(rval)) {
            raise_errmsg(ERR_ARRAY_DELIMITER, pystr, idx);
        } else {
            raise_errmsg(ERR_ARRAY_VALUE_FIRST, pystr, idx);
        }
        goto bail;
    }
    *next_idx_ptr = idx + 1;
    return rval;
bail:
    Py_XDECREF(val);
    Py_DECREF(rval);
    return NULL;
}
#endif /* PY_MAJOR_VERSION < 3 */

static PyObject *
_parse_array_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read a JSON array from PyString pystr.
    idx is the index of the first character after the opening brace.
    *next_idx_ptr is a return-by-reference index to the first character after
        the closing brace.

    Returns a new PyList
    */
    PY2_UNUSED int kind = PyUnicode_KIND(pystr);
    void *str = PyUnicode_DATA(pystr);
    Py_ssize_t end_idx = PyUnicode_GET_LENGTH(pystr) - 1;
    PyObject *val = NULL;
    PyObject *rval = PyList_New(0);
    Py_ssize_t next_idx;
    if (rval == NULL)
        return NULL;

    /* skip whitespace after [ */
    while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;

    /* only loop if the array is non-empty */
    if (idx <= end_idx && PyUnicode_READ(kind, str, idx) != ']') {
        int trailing_delimiter = 0;
        while (idx <= end_idx) {
            trailing_delimiter = 0;
            /* read any JSON term  */
            val = scan_once_unicode(s, pystr, idx, &next_idx);
            if (val == NULL) {
                goto bail;
            }

            if (PyList_Append(rval, val) == -1)
                goto bail;

            Py_CLEAR(val);
            idx = next_idx;

            /* skip whitespace between term and , */
            while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;

            /* bail if the array is closed or we didn't get the , delimiter */
            if (idx > end_idx) break;
            if (PyUnicode_READ(kind, str, idx) == ']') {
                break;
            }
            else if (PyUnicode_READ(kind, str, idx) != ',') {
                raise_errmsg(ERR_ARRAY_DELIMITER, pystr, idx);
                goto bail;
            }
            idx++;

            /* skip whitespace after , */
            while (idx <= end_idx && IS_WHITESPACE(PyUnicode_READ(kind, str, idx))) idx++;
            trailing_delimiter = 1;
        }
        if (trailing_delimiter) {
            raise_errmsg(ERR_EXPECTING_VALUE, pystr, idx);
            goto bail;
        }
    }

    /* verify that idx < end_idx, str[idx] should be ']' */
    if (idx > end_idx || PyUnicode_READ(kind, str, idx) != ']') {
        if (PyList_GET_SIZE(rval)) {
            raise_errmsg(ERR_ARRAY_DELIMITER, pystr, idx);
        } else {
            raise_errmsg(ERR_ARRAY_VALUE_FIRST, pystr, idx);
        }
        goto bail;
    }
    *next_idx_ptr = idx + 1;
    return rval;
bail:
    Py_XDECREF(val);
    Py_DECREF(rval);
    return NULL;
}

static PyObject *
_parse_constant(PyScannerObject *s, PyObject *pystr, PyObject *constant, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read a JSON constant from PyString pystr.
    constant is the Python string that was found
        ("NaN", "Infinity", "-Infinity").
    idx is the index of the first character of the constant
    *next_idx_ptr is a return-by-reference index to the first character after
        the constant.

    Returns the result of parse_constant
    */
    PyObject *rval;
    if (s->parse_constant == Py_None) {
        raise_errmsg(ERR_EXPECTING_VALUE, pystr, idx);
        return NULL;
    }

    /* rval = parse_constant(constant) */
    rval = PyObject_CallOneArg(s->parse_constant, constant);
    idx += PyString_GET_SIZE(constant);
    *next_idx_ptr = idx;
    return rval;
}

#if PY_MAJOR_VERSION < 3
static PyObject *
_match_number_str(PyScannerObject *s, PyObject *pystr, Py_ssize_t start, Py_ssize_t *next_idx_ptr)
{
    /* Read a JSON number from PyString pystr.
    idx is the index of the first character of the number
    *next_idx_ptr is a return-by-reference index to the first character after
        the number.

    Returns a new PyObject representation of that number:
        PyInt, PyLong, or PyFloat.
        May return other types if parse_int or parse_float are set
    */
    char *str = PyString_AS_STRING(pystr);
    Py_ssize_t end_idx = PyString_GET_SIZE(pystr) - 1;
    Py_ssize_t idx = start;
    int is_float = 0;
    PyObject *rval;
    PyObject *numstr;

    /* read a sign if it's there, make sure it's not the end of the string */
    if (str[idx] == '-') {
        if (idx >= end_idx) {
            raise_errmsg(ERR_EXPECTING_VALUE, pystr, start);
            return NULL;
        }
        idx++;
    }

    /* read as many integer digits as we find as long as it doesn't start with 0 */
    if (str[idx] >= '1' && str[idx] <= '9') {
        idx++;
        while (idx <= end_idx && str[idx] >= '0' && str[idx] <= '9') idx++;
    }
    /* if it starts with 0 we only expect one integer digit */
    else if (str[idx] == '0') {
        idx++;
    }
    /* no integer digits, error */
    else {
        raise_errmsg(ERR_EXPECTING_VALUE, pystr, start);
        return NULL;
    }

    /* if the next char is '.' followed by a digit then read all float digits */
    if (idx < end_idx && str[idx] == '.' && str[idx + 1] >= '0' && str[idx + 1] <= '9') {
        is_float = 1;
        idx += 2;
        while (idx <= end_idx && str[idx] >= '0' && str[idx] <= '9') idx++;
    }

    /* if the next char is 'e' or 'E' then maybe read the exponent (or backtrack) */
    if (idx < end_idx && (str[idx] == 'e' || str[idx] == 'E')) {

        /* save the index of the 'e' or 'E' just in case we need to backtrack */
        Py_ssize_t e_start = idx;
        idx++;

        /* read an exponent sign if present */
        if (idx < end_idx && (str[idx] == '-' || str[idx] == '+')) idx++;

        /* read all digits */
        while (idx <= end_idx && str[idx] >= '0' && str[idx] <= '9') idx++;

        /* if we got a digit, then parse as float. if not, backtrack */
        if (str[idx - 1] >= '0' && str[idx - 1] <= '9') {
            is_float = 1;
        }
        else {
            idx = e_start;
        }
    }

    /* copy the section we determined to be a number */
    numstr = PyString_FromStringAndSize(&str[start], idx - start);
    if (numstr == NULL)
        return NULL;
    if (is_float) {
        /* parse as a float using a fast path if available, otherwise call user defined method */
        if (s->parse_float != (PyObject *)&PyFloat_Type) {
            rval = PyObject_CallOneArg(s->parse_float, numstr);
        }
        else {
            /* rval = PyFloat_FromDouble(PyOS_ascii_atof(PyString_AS_STRING(numstr))); */
            double d = PyOS_string_to_double(PyString_AS_STRING(numstr),
                                             NULL, NULL);
            if (d == -1.0 && PyErr_Occurred()) {
                Py_DECREF(numstr);
                return NULL;
            }
            rval = PyFloat_FromDouble(d);
        }
    }
    else {
        /* parse as an int using a fast path if available, otherwise call user defined method */
        if (s->parse_int != (PyObject *)&PyInt_Type) {
            rval = PyObject_CallOneArg(s->parse_int, numstr);
        }
        else {
            rval = PyInt_FromString(PyString_AS_STRING(numstr), NULL, 10);
        }
    }
    Py_DECREF(numstr);
    *next_idx_ptr = idx;
    return rval;
}
#endif /* PY_MAJOR_VERSION < 3 */

static PyObject *
_match_number_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t start, Py_ssize_t *next_idx_ptr)
{
    /* Read a JSON number from PyUnicode pystr.
    idx is the index of the first character of the number
    *next_idx_ptr is a return-by-reference index to the first character after
        the number.

    Returns a new PyObject representation of that number:
        PyInt, PyLong, or PyFloat.
        May return other types if parse_int or parse_float are set
    */
    PY2_UNUSED int kind = PyUnicode_KIND(pystr);
    void *str = PyUnicode_DATA(pystr);
    Py_ssize_t end_idx = PyUnicode_GET_LENGTH(pystr) - 1;
    Py_ssize_t idx = start;
    int is_float = 0;
    JSON_UNICHR c;
    PyObject *rval;
    PyObject *numstr;

    /* read a sign if it's there, make sure it's not the end of the string */
    if (PyUnicode_READ(kind, str, idx) == '-') {
        if (idx >= end_idx) {
            raise_errmsg(ERR_EXPECTING_VALUE, pystr, start);
            return NULL;
        }
        idx++;
    }

    /* read as many integer digits as we find as long as it doesn't start with 0 */
    c = PyUnicode_READ(kind, str, idx);
    if (c == '0') {
        /* if it starts with 0 we only expect one integer digit */
        idx++;
    }
    else if (IS_DIGIT(c)) {
        idx++;
        while (idx <= end_idx && IS_DIGIT(PyUnicode_READ(kind, str, idx))) {
            idx++;
        }
    }
    else {
        /* no integer digits, error */
        raise_errmsg(ERR_EXPECTING_VALUE, pystr, start);
        return NULL;
    }

    /* if the next char is '.' followed by a digit then read all float digits */
    if (idx < end_idx &&
        PyUnicode_READ(kind, str, idx) == '.' &&
        IS_DIGIT(PyUnicode_READ(kind, str, idx + 1))) {
        is_float = 1;
        idx += 2;
        while (idx <= end_idx && IS_DIGIT(PyUnicode_READ(kind, str, idx))) idx++;
    }

    /* if the next char is 'e' or 'E' then maybe read the exponent (or backtrack) */
    if (idx < end_idx &&
        (PyUnicode_READ(kind, str, idx) == 'e' ||
            PyUnicode_READ(kind, str, idx) == 'E')) {
        Py_ssize_t e_start = idx;
        idx++;

        /* read an exponent sign if present */
        if (idx < end_idx &&
            (PyUnicode_READ(kind, str, idx) == '-' ||
                PyUnicode_READ(kind, str, idx) == '+')) idx++;

        /* read all digits */
        while (idx <= end_idx && IS_DIGIT(PyUnicode_READ(kind, str, idx))) idx++;

        /* if we got a digit, then parse as float. if not, backtrack */
        if (IS_DIGIT(PyUnicode_READ(kind, str, idx - 1))) {
            is_float = 1;
        }
        else {
            idx = e_start;
        }
    }

    /* copy the section we determined to be a number */
#if PY_MAJOR_VERSION >= 3
    numstr = PyUnicode_Substring(pystr, start, idx);
#else
    numstr = PyUnicode_FromUnicode(&((Py_UNICODE *)str)[start], idx - start);
#endif
    if (numstr == NULL)
        return NULL;
    if (is_float) {
        /* parse as a float using a fast path if available, otherwise call user defined method */
        if (s->parse_float != (PyObject *)&PyFloat_Type) {
            rval = PyObject_CallOneArg(s->parse_float, numstr);
        }
        else {
#if PY_MAJOR_VERSION >= 3
            rval = PyFloat_FromString(numstr);
#else
            rval = PyFloat_FromString(numstr, NULL);
#endif
        }
    }
    else {
        /* no fast path for unicode -> int, just call */
        rval = PyObject_CallOneArg(s->parse_int, numstr);
    }
    Py_DECREF(numstr);
    *next_idx_ptr = idx;
    return rval;
}

#if PY_MAJOR_VERSION < 3
static PyObject *
scan_once_str(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read one JSON term (of any kind) from PyString pystr.
    idx is the index of the first character of the term
    *next_idx_ptr is a return-by-reference index to the first character after
        the number.

    Returns a new PyObject representation of the term.
    */
    char *str = PyString_AS_STRING(pystr);
    Py_ssize_t length = PyString_GET_SIZE(pystr);
    PyObject *rval = NULL;
    int fallthrough = 0;
    if (idx < 0 || idx >= length) {
        raise_errmsg(ERR_EXPECTING_VALUE, pystr, idx);
        return NULL;
    }
    switch (str[idx]) {
        case '"':
            /* string */
            rval = scanstring_str(pystr, idx + 1,
                PyString_AS_STRING(s->encoding),
                s->strict,
                next_idx_ptr);
            break;
        case '{':
            /* object */
            if (Py_EnterRecursiveCall(" while decoding a JSON object "
                                      "from a string"))
                return NULL;
            rval = _parse_object_str(s, pystr, idx + 1, next_idx_ptr);
            Py_LeaveRecursiveCall();
            break;
        case '[':
            /* array */
            if (Py_EnterRecursiveCall(" while decoding a JSON array "
                                      "from a string"))
                return NULL;
            rval = _parse_array_str(s, pystr, idx + 1, next_idx_ptr);
            Py_LeaveRecursiveCall();
            break;
        case 'n':
            /* null */
            if ((idx + 3 < length) && str[idx + 1] == 'u' && str[idx + 2] == 'l' && str[idx + 3] == 'l') {
                Py_INCREF(Py_None);
                *next_idx_ptr = idx + 4;
                rval = Py_None;
            }
            else
                fallthrough = 1;
            break;
        case 't':
            /* true */
            if ((idx + 3 < length) && str[idx + 1] == 'r' && str[idx + 2] == 'u' && str[idx + 3] == 'e') {
                Py_INCREF(Py_True);
                *next_idx_ptr = idx + 4;
                rval = Py_True;
            }
            else
                fallthrough = 1;
            break;
        case 'f':
            /* false */
            if ((idx + 4 < length) && str[idx + 1] == 'a' && str[idx + 2] == 'l' && str[idx + 3] == 's' && str[idx + 4] == 'e') {
                Py_INCREF(Py_False);
                *next_idx_ptr = idx + 5;
                rval = Py_False;
            }
            else
                fallthrough = 1;
            break;
        case 'N':
            /* NaN */
            if ((idx + 2 < length) && str[idx + 1] == 'a' && str[idx + 2] == 'N') {
                rval = _parse_constant(s, pystr, JSON_NaN, idx, next_idx_ptr);
            }
            else
                fallthrough = 1;
            break;
        case 'I':
            /* Infinity */
            if ((idx + 7 < length) && str[idx + 1] == 'n' && str[idx + 2] == 'f' && str[idx + 3] == 'i' && str[idx + 4] == 'n' && str[idx + 5] == 'i' && str[idx + 6] == 't' && str[idx + 7] == 'y') {
                rval = _parse_constant(s, pystr, JSON_Infinity, idx, next_idx_ptr);
            }
            else
                fallthrough = 1;
            break;
        case '-':
            /* -Infinity */
            if ((idx + 8 < length) && str[idx + 1] == 'I' && str[idx + 2] == 'n' && str[idx + 3] == 'f' && str[idx + 4] == 'i' && str[idx + 5] == 'n' && str[idx + 6] == 'i' && str[idx + 7] == 't' && str[idx + 8] == 'y') {
                rval = _parse_constant(s, pystr, JSON_NegInfinity, idx, next_idx_ptr);
            }
            else
                fallthrough = 1;
            break;
        default:
            fallthrough = 1;
    }
    /* Didn't find a string, object, array, or named constant. Look for a number. */
    if (fallthrough)
        rval = _match_number_str(s, pystr, idx, next_idx_ptr);
    return rval;
}
#endif /* PY_MAJOR_VERSION < 3 */


static PyObject *
scan_once_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read one JSON term (of any kind) from PyUnicode pystr.
    idx is the index of the first character of the term
    *next_idx_ptr is a return-by-reference index to the first character after
        the number.

    Returns a new PyObject representation of the term.
    */
    PY2_UNUSED int kind = PyUnicode_KIND(pystr);
    void *str = PyUnicode_DATA(pystr);
    Py_ssize_t length = PyUnicode_GET_LENGTH(pystr);
    PyObject *rval = NULL;
    int fallthrough = 0;
    if (idx < 0 || idx >= length) {
        raise_errmsg(ERR_EXPECTING_VALUE, pystr, idx);
        return NULL;
    }
    switch (PyUnicode_READ(kind, str, idx)) {
        case '"':
            /* string */
            rval = scanstring_unicode(pystr, idx + 1,
                s->strict,
                next_idx_ptr);
            break;
        case '{':
            /* object */
            if (Py_EnterRecursiveCall(" while decoding a JSON object "
                                      "from a unicode string"))
                return NULL;
            rval = _parse_object_unicode(s, pystr, idx + 1, next_idx_ptr);
            Py_LeaveRecursiveCall();
            break;
        case '[':
            /* array */
            if (Py_EnterRecursiveCall(" while decoding a JSON array "
                                      "from a unicode string"))
                return NULL;
            rval = _parse_array_unicode(s, pystr, idx + 1, next_idx_ptr);
            Py_LeaveRecursiveCall();
            break;
        case 'n':
            /* null */
            if ((idx + 3 < length) &&
                PyUnicode_READ(kind, str, idx + 1) == 'u' &&
                PyUnicode_READ(kind, str, idx + 2) == 'l' &&
                PyUnicode_READ(kind, str, idx + 3) == 'l') {
                Py_INCREF(Py_None);
                *next_idx_ptr = idx + 4;
                rval = Py_None;
            }
            else
                fallthrough = 1;
            break;
        case 't':
            /* true */
            if ((idx + 3 < length) &&
                PyUnicode_READ(kind, str, idx + 1) == 'r' &&
                PyUnicode_READ(kind, str, idx + 2) == 'u' &&
                PyUnicode_READ(kind, str, idx + 3) == 'e') {
                Py_INCREF(Py_True);
                *next_idx_ptr = idx + 4;
                rval = Py_True;
            }
            else
                fallthrough = 1;
            break;
        case 'f':
            /* false */
            if ((idx + 4 < length) &&
                PyUnicode_READ(kind, str, idx + 1) == 'a' &&
                PyUnicode_READ(kind, str, idx + 2) == 'l' &&
                PyUnicode_READ(kind, str, idx + 3) == 's' &&
                PyUnicode_READ(kind, str, idx + 4) == 'e') {
                Py_INCREF(Py_False);
                *next_idx_ptr = idx + 5;
                rval = Py_False;
            }
            else
                fallthrough = 1;
            break;
        case 'N':
            /* NaN */
            if ((idx + 2 < length) &&
                PyUnicode_READ(kind, str, idx + 1) == 'a' &&
                PyUnicode_READ(kind, str, idx + 2) == 'N') {
                rval = _parse_constant(s, pystr, JSON_NaN, idx, next_idx_ptr);
            }
            else
                fallthrough = 1;
            break;
        case 'I':
            /* Infinity */
            if ((idx + 7 < length) &&
                PyUnicode_READ(kind, str, idx + 1) == 'n' &&
                PyUnicode_READ(kind, str, idx + 2) == 'f' &&
                PyUnicode_READ(kind, str, idx + 3) == 'i' &&
                PyUnicode_READ(kind, str, idx + 4) == 'n' &&
                PyUnicode_READ(kind, str, idx + 5) == 'i' &&
                PyUnicode_READ(kind, str, idx + 6) == 't' &&
                PyUnicode_READ(kind, str, idx + 7) == 'y') {
                rval = _parse_constant(s, pystr, JSON_Infinity, idx, next_idx_ptr);
            }
            else
                fallthrough = 1;
            break;
        case '-':
            /* -Infinity */
            if ((idx + 8 < length) &&
                PyUnicode_READ(kind, str, idx + 1) == 'I' &&
                PyUnicode_READ(kind, str, idx + 2) == 'n' &&
                PyUnicode_READ(kind, str, idx + 3) == 'f' &&
                PyUnicode_READ(kind, str, idx + 4) == 'i' &&
                PyUnicode_READ(kind, str, idx + 5) == 'n' &&
                PyUnicode_READ(kind, str, idx + 6) == 'i' &&
                PyUnicode_READ(kind, str, idx + 7) == 't' &&
                PyUnicode_READ(kind, str, idx + 8) == 'y') {
                rval = _parse_constant(s, pystr, JSON_NegInfinity, idx, next_idx_ptr);
            }
            else
                fallthrough = 1;
            break;
        default:
            fallthrough = 1;
    }
    /* Didn't find a string, object, array, or named constant. Look for a number. */
    if (fallthrough)
        rval = _match_number_unicode(s, pystr, idx, next_idx_ptr);
    return rval;
}

static PyObject *
scanner_call(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* Python callable interface to scan_once_{str,unicode} */
    PyObject *pystr;
    PyObject *rval;
    Py_ssize_t idx;
    Py_ssize_t next_idx = -1;
    static char *kwlist[] = {"string", "idx", NULL};
    PyScannerObject *s;
    assert(PyScanner_Check(self));
    s = (PyScannerObject *)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&:scan_once", kwlist, &pystr, _convertPyInt_AsSsize_t, &idx))
        return NULL;

    if (PyUnicode_Check(pystr)) {
        if (PyUnicode_READY(pystr))
            return NULL;
        rval = scan_once_unicode(s, pystr, idx, &next_idx);
    }
#if PY_MAJOR_VERSION < 3
    else if (PyString_Check(pystr)) {
        rval = scan_once_str(s, pystr, idx, &next_idx);
    }
#endif /* PY_MAJOR_VERSION < 3 */
    else {
        PyErr_Format(PyExc_TypeError,
                 "first argument must be a string, not %.80s",
                 Py_TYPE(pystr)->tp_name);
        return NULL;
    }
    PyDict_Clear(s->memo);
    return _build_rval_index_tuple(rval, next_idx);
}

static PyObject *
JSON_ParseEncoding(PyObject *encoding)
{
    if (encoding == Py_None)
        return JSON_InternFromString(DEFAULT_ENCODING);
#if PY_MAJOR_VERSION >= 3
    if (PyUnicode_Check(encoding)) {
        if (PyUnicode_AsUTF8(encoding) == NULL) {
            return NULL;
        }
        Py_INCREF(encoding);
        return encoding;
    }
#else /* PY_MAJOR_VERSION >= 3 */
    if (PyString_Check(encoding)) {
        Py_INCREF(encoding);
        return encoding;
    }
    if (PyUnicode_Check(encoding))
        return PyUnicode_AsEncodedString(encoding, NULL, NULL);
#endif /* PY_MAJOR_VERSION >= 3 */
    PyErr_SetString(PyExc_TypeError, "encoding must be a string");
    return NULL;
}

static PyObject *
scanner_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    /* Initialize Scanner object */
    PyObject *ctx;
    static char *kwlist[] = {"context", NULL};
    PyScannerObject *s;
    PyObject *encoding;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:make_scanner", kwlist, &ctx))
        return NULL;

    s = (PyScannerObject *)type->tp_alloc(type, 0);
    if (s == NULL)
        return NULL;

    if (s->memo == NULL) {
        s->memo = PyDict_New();
        if (s->memo == NULL)
            goto bail;
    }

    encoding = PyObject_GetAttrString(ctx, "encoding");
    if (encoding == NULL)
        goto bail;
    s->encoding = JSON_ParseEncoding(encoding);
    Py_XDECREF(encoding);
    if (s->encoding == NULL)
        goto bail;

    /* All of these will fail "gracefully" so we don't need to verify them */
    s->strict_bool = PyObject_GetAttrString(ctx, "strict");
    if (s->strict_bool == NULL)
        goto bail;
    s->strict = PyObject_IsTrue(s->strict_bool);
    if (s->strict < 0)
        goto bail;
    s->object_hook = PyObject_GetAttrString(ctx, "object_hook");
    if (s->object_hook == NULL)
        goto bail;
    s->pairs_hook = PyObject_GetAttrString(ctx, "object_pairs_hook");
    if (s->pairs_hook == NULL)
        goto bail;
    s->parse_float = PyObject_GetAttrString(ctx, "parse_float");
    if (s->parse_float == NULL)
        goto bail;
    s->parse_int = PyObject_GetAttrString(ctx, "parse_int");
    if (s->parse_int == NULL)
        goto bail;
    s->parse_constant = PyObject_GetAttrString(ctx, "parse_constant");
    if (s->parse_constant == NULL)
        goto bail;

    return (PyObject *)s;

bail:
    Py_DECREF(s);
    return NULL;
}

PyDoc_STRVAR(scanner_doc, "JSON scanner object");

static
PyTypeObject PyScannerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "simplejson._speedups.Scanner",       /* tp_name */
    sizeof(PyScannerObject), /* tp_basicsize */
    0,                    /* tp_itemsize */
    scanner_dealloc, /* tp_dealloc */
    0,                    /* tp_print */
    0,                    /* tp_getattr */
    0,                    /* tp_setattr */
    0,                    /* tp_compare */
    0,                    /* tp_repr */
    0,                    /* tp_as_number */
    0,                    /* tp_as_sequence */
    0,                    /* tp_as_mapping */
    0,                    /* tp_hash */
    scanner_call,         /* tp_call */
    0,                    /* tp_str */
    0,/* PyObject_GenericGetAttr, */                    /* tp_getattro */
    0,/* PyObject_GenericSetAttr, */                    /* tp_setattro */
    0,                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,   /* tp_flags */
    scanner_doc,          /* tp_doc */
    scanner_traverse,                    /* tp_traverse */
    scanner_clear,                    /* tp_clear */
    0,                    /* tp_richcompare */
    0,                    /* tp_weaklistoffset */
    0,                    /* tp_iter */
    0,                    /* tp_iternext */
    0,                    /* tp_methods */
    scanner_members,                    /* tp_members */
    0,                    /* tp_getset */
    0,                    /* tp_base */
    0,                    /* tp_dict */
    0,                    /* tp_descr_get */
    0,                    /* tp_descr_set */
    0,                    /* tp_dictoffset */
    0,                    /* tp_init */
    0,/* PyType_GenericAlloc, */        /* tp_alloc */
    scanner_new,          /* tp_new */
    0,/* PyObject_GC_Del, */              /* tp_free */
};

static PyObject *
encoder_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {
        "markers",
        "default",
        "encoder",
        "indent",
        "key_separator",
        "item_separator",
        "sort_keys",
        "skipkeys",
        "allow_nan",
        "key_memo",
        "use_decimal",
        "namedtuple_as_object",
        "tuple_as_array",
        "int_as_string_bitcount",
        "item_sort_key",
        "encoding",
        "for_json",
        "ignore_nan",
        "Decimal",
        "iterable_as_array",
        NULL};

    PyEncoderObject *s;
    PyObject *markers, *defaultfn, *encoder, *indent, *key_separator;
    PyObject *item_separator, *sort_keys, *skipkeys, *allow_nan, *key_memo;
    PyObject *use_decimal, *namedtuple_as_object, *tuple_as_array, *iterable_as_array;
    PyObject *int_as_string_bitcount, *item_sort_key, *encoding, *for_json;
    PyObject *ignore_nan, *Decimal;
    int is_true;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOOOOOOOOOOOOO:make_encoder", kwlist,
        &markers, &defaultfn, &encoder, &indent, &key_separator, &item_separator,
        &sort_keys, &skipkeys, &allow_nan, &key_memo, &use_decimal,
        &namedtuple_as_object, &tuple_as_array,
        &int_as_string_bitcount, &item_sort_key, &encoding, &for_json,
        &ignore_nan, &Decimal, &iterable_as_array))
        return NULL;

    s = (PyEncoderObject *)type->tp_alloc(type, 0);
    if (s == NULL)
        return NULL;

    Py_INCREF(markers);
    s->markers = markers;
    Py_INCREF(defaultfn);
    s->defaultfn = defaultfn;
    Py_INCREF(encoder);
    s->encoder = encoder;
#if PY_MAJOR_VERSION >= 3
    if (encoding == Py_None) {
        s->encoding = NULL;
    }
    else
#endif /* PY_MAJOR_VERSION >= 3 */
    {
        s->encoding = JSON_ParseEncoding(encoding);
        if (s->encoding == NULL)
            goto bail;
    }
    Py_INCREF(indent);
    s->indent = indent;
    Py_INCREF(key_separator);
    s->key_separator = key_separator;
    Py_INCREF(item_separator);
    s->item_separator = item_separator;
    Py_INCREF(skipkeys);
    s->skipkeys_bool = skipkeys;
    s->skipkeys = PyObject_IsTrue(skipkeys);
    if (s->skipkeys < 0)
        goto bail;
    Py_INCREF(key_memo);
    s->key_memo = key_memo;
    s->fast_encode = (PyCFunction_Check(s->encoder) && PyCFunction_GetFunction(s->encoder) == (PyCFunction)py_encode_basestring_ascii);
    is_true = PyObject_IsTrue(ignore_nan);
    if (is_true < 0)
        goto bail;
    s->allow_or_ignore_nan = is_true ? JSON_IGNORE_NAN : 0;
    is_true = PyObject_IsTrue(allow_nan);
    if (is_true < 0)
        goto bail;
    s->allow_or_ignore_nan |= is_true ? JSON_ALLOW_NAN : 0;
    s->use_decimal = PyObject_IsTrue(use_decimal);
    if (s->use_decimal < 0)
        goto bail;
    s->namedtuple_as_object = PyObject_IsTrue(namedtuple_as_object);
    if (s->namedtuple_as_object < 0)
        goto bail;
    s->tuple_as_array = PyObject_IsTrue(tuple_as_array);
    if (s->tuple_as_array < 0)
        goto bail;
    s->iterable_as_array = PyObject_IsTrue(iterable_as_array);
    if (s->iterable_as_array < 0)
        goto bail;
    if (PyInt_Check(int_as_string_bitcount) || PyLong_Check(int_as_string_bitcount)) {
        static const unsigned long long_long_bitsize = SIZEOF_LONG_LONG * 8;
        long int_as_string_bitcount_val = PyLong_AsLong(int_as_string_bitcount);
        if (int_as_string_bitcount_val > 0 && int_as_string_bitcount_val < (long)long_long_bitsize) {
            s->max_long_size = PyLong_FromUnsignedLongLong(1ULL << (int)int_as_string_bitcount_val);
            s->min_long_size = PyLong_FromLongLong(-1LL << (int)int_as_string_bitcount_val);
            if (s->min_long_size == NULL || s->max_long_size == NULL) {
                goto bail;
            }
        }
        else {
            PyErr_Format(PyExc_TypeError,
                         "int_as_string_bitcount (%ld) must be greater than 0 and less than the number of bits of a `long long` type (%lu bits)",
                         int_as_string_bitcount_val, long_long_bitsize);
            goto bail;
        }
    }
    else if (int_as_string_bitcount == Py_None) {
        Py_INCREF(Py_None);
        s->max_long_size = Py_None;
        Py_INCREF(Py_None);
        s->min_long_size = Py_None;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "int_as_string_bitcount must be None or an integer");
        goto bail;
    }
    if (item_sort_key != Py_None) {
        if (!PyCallable_Check(item_sort_key)) {
            PyErr_SetString(PyExc_TypeError, "item_sort_key must be None or callable");
            goto bail;
        }
    }
    else {
        is_true = PyObject_IsTrue(sort_keys);
        if (is_true < 0)
            goto bail;
        if (is_true) {
            static PyObject *itemgetter0 = NULL;
            if (!itemgetter0) {
                PyObject *operator = PyImport_ImportModule("operator");
                if (!operator)
                    goto bail;
                itemgetter0 = PyObject_CallMethod(operator, "itemgetter", "i", 0);
                Py_DECREF(operator);
            }
            item_sort_key = itemgetter0;
            if (!item_sort_key)
                goto bail;
        }
    }
    if (item_sort_key == Py_None) {
        Py_INCREF(Py_None);
        s->item_sort_kw = Py_None;
    }
    else {
        s->item_sort_kw = PyDict_New();
        if (s->item_sort_kw == NULL)
            goto bail;
        if (PyDict_SetItemString(s->item_sort_kw, "key", item_sort_key))
            goto bail;
    }
    Py_INCREF(sort_keys);
    s->sort_keys = sort_keys;
    Py_INCREF(item_sort_key);
    s->item_sort_key = item_sort_key;
    Py_INCREF(Decimal);
    s->Decimal = Decimal;
    s->for_json = PyObject_IsTrue(for_json);
    if (s->for_json < 0)
        goto bail;

    return (PyObject *)s;

bail:
    Py_DECREF(s);
    return NULL;
}

static PyObject *
encoder_call(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* Python callable interface to encode_listencode_obj */
    static char *kwlist[] = {"obj", "_current_indent_level", NULL};
    PyObject *obj;
    Py_ssize_t indent_level;
    PyEncoderObject *s;
    JSON_Accu rval;
    assert(PyEncoder_Check(self));
    s = (PyEncoderObject *)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&:_iterencode", kwlist,
        &obj, _convertPyInt_AsSsize_t, &indent_level))
        return NULL;
    if (JSON_Accu_Init(&rval))
        return NULL;
    if (encoder_listencode_obj(s, &rval, obj, indent_level)) {
        JSON_Accu_Destroy(&rval);
        return NULL;
    }
    return JSON_Accu_FinishAsList(&rval);
}

static PyObject *
_encoded_const(PyObject *obj)
{
    /* Return the JSON string representation of None, True, False */
    if (obj == Py_None) {
        static PyObject *s_null = NULL;
        if (s_null == NULL) {
            s_null = JSON_InternFromString("null");
        }
        Py_INCREF(s_null);
        return s_null;
    }
    else if (obj == Py_True) {
        static PyObject *s_true = NULL;
        if (s_true == NULL) {
            s_true = JSON_InternFromString("true");
        }
        Py_INCREF(s_true);
        return s_true;
    }
    else if (obj == Py_False) {
        static PyObject *s_false = NULL;
        if (s_false == NULL) {
            s_false = JSON_InternFromString("false");
        }
        Py_INCREF(s_false);
        return s_false;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "not a const");
        return NULL;
    }
}

static PyObject *
encoder_encode_float(PyEncoderObject *s, PyObject *obj)
{
    /* Return the JSON representation of a PyFloat */
    double i = PyFloat_AS_DOUBLE(obj);
    if (!Py_IS_FINITE(i)) {
        if (!s->allow_or_ignore_nan) {
            PyErr_SetString(PyExc_ValueError, "Out of range float values are not JSON compliant");
            return NULL;
        }
        if (s->allow_or_ignore_nan & JSON_IGNORE_NAN) {
            return _encoded_const(Py_None);
        }
        /* JSON_ALLOW_NAN is set */
        else if (i > 0) {
            Py_INCREF(JSON_Infinity);
            return JSON_Infinity;
        }
        else if (i < 0) {
            Py_INCREF(JSON_NegInfinity);
            return JSON_NegInfinity;
        }
        else {
            Py_INCREF(JSON_NaN);
            return JSON_NaN;
        }
    }
    /* Use a better float format here? */
    if (PyFloat_CheckExact(obj)) {
        return PyObject_Repr(obj);
    }
    else {
        /* See #118, do not trust custom str/repr */
        PyObject *res;
        PyObject *tmp = PyObject_CallOneArg((PyObject *)&PyFloat_Type, obj);
        if (tmp == NULL) {
            return NULL;
        }
        res = PyObject_Repr(tmp);
        Py_DECREF(tmp);
        return res;
    }
}

static PyObject *
encoder_encode_string(PyEncoderObject *s, PyObject *obj)
{
    /* Return the JSON representation of a string */
    PyObject *encoded;

    if (s->fast_encode) {
        return py_encode_basestring_ascii(NULL, obj);
    }
    encoded = PyObject_CallOneArg(s->encoder, obj);
    if (encoded != NULL &&
#if PY_MAJOR_VERSION < 3
        !PyString_Check(encoded) &&
#endif /* PY_MAJOR_VERSION < 3 */
        !PyUnicode_Check(encoded))
    {
        PyErr_Format(PyExc_TypeError,
                     "encoder() must return a string, not %.80s",
                     Py_TYPE(encoded)->tp_name);
        Py_DECREF(encoded);
        return NULL;
    }
    return encoded;
}

static int
_steal_accumulate(JSON_Accu *accu, PyObject *stolen)
{
    /* Append stolen and then decrement its reference count */
    int rval = JSON_Accu_Accumulate(accu, stolen);
    Py_DECREF(stolen);
    return rval;
}

static int
encoder_listencode_obj(PyEncoderObject *s, JSON_Accu *rval, PyObject *obj, Py_ssize_t indent_level)
{
    /* Encode Python object obj to a JSON term, rval is a PyList */
    int rv = -1;
    do {
        PyObject *newobj;
        if (obj == Py_None || obj == Py_True || obj == Py_False) {
            PyObject *cstr = _encoded_const(obj);
            if (cstr != NULL)
                rv = _steal_accumulate(rval, cstr);
        }
        else if ((PyBytes_Check(obj) && s->encoding != NULL) ||
                 PyUnicode_Check(obj))
        {
            PyObject *encoded = encoder_encode_string(s, obj);
            if (encoded != NULL)
                rv = _steal_accumulate(rval, encoded);
        }
        else if (PyInt_Check(obj) || PyLong_Check(obj)) {
            PyObject *encoded;
            if (PyInt_CheckExact(obj) || PyLong_CheckExact(obj)) {
                encoded = PyObject_Str(obj);
            }
            else {
                /* See #118, do not trust custom str/repr */
                PyObject *tmp = PyObject_CallOneArg((PyObject *)&PyLong_Type, obj);
                if (tmp == NULL) {
                    encoded = NULL;
                }
                else {
                    encoded = PyObject_Str(tmp);
                    Py_DECREF(tmp);
                }
            }
            if (encoded != NULL) {
                encoded = maybe_quote_bigint(s, encoded, obj);
                if (encoded == NULL)
                    break;
                rv = _steal_accumulate(rval, encoded);
            }
        }
        else if (PyFloat_Check(obj)) {
            PyObject *encoded = encoder_encode_float(s, obj);
            if (encoded != NULL)
                rv = _steal_accumulate(rval, encoded);
        }
        else if (s->for_json && _call_json_method(obj, FOR_JSON_METHOD_NAME, &newobj)) {
            if (newobj == NULL) {
                return -1;
            }
            if (Py_EnterRecursiveCall(" while encoding a JSON object")) {
                Py_DECREF(newobj);
                return rv;
            }
            rv = encoder_listencode_obj(s, rval, newobj, indent_level);
            Py_DECREF(newobj);
            Py_LeaveRecursiveCall();
        }
        else if (s->namedtuple_as_object && _call_json_method(obj, ASDICT_METHOD_NAME, &newobj)) {
            if (newobj == NULL) {
                return -1;
            }
            if (Py_EnterRecursiveCall(" while encoding a JSON object")) {
                Py_DECREF(newobj);
                return rv;
            }
            if (PyDict_Check(newobj)) {
                rv = encoder_listencode_dict(s, rval, newobj, indent_level);
            } else {
                PyErr_Format(
                    PyExc_TypeError,
                    "_asdict() must return a dict, not %.80s",
                    Py_TYPE(newobj)->tp_name
                );
                rv = -1;
            }
            Py_DECREF(newobj);
            Py_LeaveRecursiveCall();
        }
        else if (PyList_Check(obj) || (s->tuple_as_array && PyTuple_Check(obj))) {
            if (Py_EnterRecursiveCall(" while encoding a JSON object"))
                return rv;
            rv = encoder_listencode_list(s, rval, obj, indent_level);
            Py_LeaveRecursiveCall();
        }
        else if (PyDict_Check(obj)) {
            if (Py_EnterRecursiveCall(" while encoding a JSON object"))
                return rv;
            rv = encoder_listencode_dict(s, rval, obj, indent_level);
            Py_LeaveRecursiveCall();
        }
        else if (s->use_decimal && PyObject_TypeCheck(obj, (PyTypeObject *)s->Decimal)) {
            PyObject *encoded = PyObject_Str(obj);
            if (encoded != NULL)
                rv = _steal_accumulate(rval, encoded);
        }
        else if (is_raw_json(obj))
        {
            PyObject *encoded = PyObject_GetAttrString(obj, "encoded_json");
            if (encoded != NULL)
                rv = _steal_accumulate(rval, encoded);
        }
        else {
            PyObject *ident = NULL;
            PyObject *newobj;
            if (s->iterable_as_array) {
                newobj = PyObject_GetIter(obj);
                if (newobj == NULL)
                    PyErr_Clear();
                else {
                    rv = encoder_listencode_list(s, rval, newobj, indent_level);
                    Py_DECREF(newobj);
                    break;
                }
            }
            if (s->markers != Py_None) {
                int has_key;
                ident = PyLong_FromVoidPtr(obj);
                if (ident == NULL)
                    break;
                has_key = PyDict_Contains(s->markers, ident);
                if (has_key) {
                    if (has_key != -1)
                        PyErr_SetString(PyExc_ValueError, "Circular reference detected");
                    Py_DECREF(ident);
                    break;
                }
                if (PyDict_SetItem(s->markers, ident, obj)) {
                    Py_DECREF(ident);
                    break;
                }
            }
            if (Py_EnterRecursiveCall(" while encoding a JSON object"))
                return rv;
            newobj = PyObject_CallOneArg(s->defaultfn, obj);
            if (newobj == NULL) {
                Py_XDECREF(ident);
                Py_LeaveRecursiveCall();
                break;
            }
            rv = encoder_listencode_obj(s, rval, newobj, indent_level);
            Py_LeaveRecursiveCall();
            Py_DECREF(newobj);
            if (rv) {
                Py_XDECREF(ident);
                rv = -1;
            }
            else if (ident != NULL) {
                if (PyDict_DelItem(s->markers, ident)) {
                    Py_XDECREF(ident);
                    rv = -1;
                }
                Py_XDECREF(ident);
            }
        }
    } while (0);
    return rv;
}

static int
encoder_listencode_dict(PyEncoderObject *s, JSON_Accu *rval, PyObject *dct, Py_ssize_t indent_level)
{
    /* Encode Python dict dct a JSON term */
    static PyObject *open_dict = NULL;
    static PyObject *close_dict = NULL;
    static PyObject *empty_dict = NULL;
    PyObject *kstr = NULL;
    PyObject *ident = NULL;
    PyObject *iter = NULL;
    PyObject *item = NULL;
    PyObject *items = NULL;
    PyObject *encoded = NULL;
    Py_ssize_t idx;

    if (open_dict == NULL || close_dict == NULL || empty_dict == NULL) {
        open_dict = JSON_InternFromString("{");
        close_dict = JSON_InternFromString("}");
        empty_dict = JSON_InternFromString("{}");
        if (open_dict == NULL || close_dict == NULL || empty_dict == NULL)
            return -1;
    }
    if (PyDict_Size(dct) == 0)
        return JSON_Accu_Accumulate(rval, empty_dict);

    if (s->markers != Py_None) {
        int has_key;
        ident = PyLong_FromVoidPtr(dct);
        if (ident == NULL)
            goto bail;
        has_key = PyDict_Contains(s->markers, ident);
        if (has_key) {
            if (has_key != -1)
                PyErr_SetString(PyExc_ValueError, "Circular reference detected");
            goto bail;
        }
        if (PyDict_SetItem(s->markers, ident, dct)) {
            goto bail;
        }
    }

    if (JSON_Accu_Accumulate(rval, open_dict))
        goto bail;

    if (s->indent != Py_None) {
        /* TODO: DOES NOT RUN */
        indent_level += 1;
        /*
            newline_indent = '\n' + (_indent * _current_indent_level)
            separator = _item_separator + newline_indent
            buf += newline_indent
        */
    }

    iter = encoder_dict_iteritems(s, dct);
    if (iter == NULL)
        goto bail;

    idx = 0;
    while ((item = PyIter_Next(iter))) {
        PyObject *encoded, *key, *value;
        if (!PyTuple_Check(item) || Py_SIZE(item) != 2) {
            PyErr_SetString(PyExc_ValueError, "items must return 2-tuples");
            goto bail;
        }
        key = PyTuple_GET_ITEM(item, 0);
        if (key == NULL)
            goto bail;
        value = PyTuple_GET_ITEM(item, 1);
        if (value == NULL)
            goto bail;

        encoded = PyDict_GetItem(s->key_memo, key);
        if (encoded != NULL) {
            Py_INCREF(encoded);
        } else {
            kstr = encoder_stringify_key(s, key);
            if (kstr == NULL)
                goto bail;
            else if (kstr == Py_None) {
                /* skipkeys */
                Py_DECREF(item);
                Py_DECREF(kstr);
                continue;
            }
        }
        if (idx) {
            if (JSON_Accu_Accumulate(rval, s->item_separator))
                goto bail;
        }
        if (encoded == NULL) {
            encoded = encoder_encode_string(s, kstr);
            Py_CLEAR(kstr);
            if (encoded == NULL)
                goto bail;
            if (PyDict_SetItem(s->key_memo, key, encoded))
                goto bail;
        }
        if (JSON_Accu_Accumulate(rval, encoded)) {
            goto bail;
        }
        Py_CLEAR(encoded);
        if (JSON_Accu_Accumulate(rval, s->key_separator))
            goto bail;
        if (encoder_listencode_obj(s, rval, value, indent_level))
            goto bail;
        Py_CLEAR(item);
        idx += 1;
    }
    Py_CLEAR(iter);
    if (PyErr_Occurred())
        goto bail;
    if (ident != NULL) {
        if (PyDict_DelItem(s->markers, ident))
            goto bail;
        Py_CLEAR(ident);
    }
    if (s->indent != Py_None) {
        /* TODO: DOES NOT RUN */
        indent_level -= 1;
        /*
            yield '\n' + (_indent * _current_indent_level)
        */
    }
    if (JSON_Accu_Accumulate(rval, close_dict))
        goto bail;
    return 0;

bail:
    Py_XDECREF(encoded);
    Py_XDECREF(items);
    Py_XDECREF(item);
    Py_XDECREF(iter);
    Py_XDECREF(kstr);
    Py_XDECREF(ident);
    return -1;
}


static int
encoder_listencode_list(PyEncoderObject *s, JSON_Accu *rval, PyObject *seq, Py_ssize_t indent_level)
{
    /* Encode Python list seq to a JSON term */
    static PyObject *open_array = NULL;
    static PyObject *close_array = NULL;
    static PyObject *empty_array = NULL;
    PyObject *ident = NULL;
    PyObject *iter = NULL;
    PyObject *obj = NULL;
    int is_true;
    int i = 0;

    if (open_array == NULL || close_array == NULL || empty_array == NULL) {
        open_array = JSON_InternFromString("[");
        close_array = JSON_InternFromString("]");
        empty_array = JSON_InternFromString("[]");
        if (open_array == NULL || close_array == NULL || empty_array == NULL)
            return -1;
    }
    ident = NULL;
    is_true = PyObject_IsTrue(seq);
    if (is_true == -1)
        return -1;
    else if (is_true == 0)
        return JSON_Accu_Accumulate(rval, empty_array);

    if (s->markers != Py_None) {
        int has_key;
        ident = PyLong_FromVoidPtr(seq);
        if (ident == NULL)
            goto bail;
        has_key = PyDict_Contains(s->markers, ident);
        if (has_key) {
            if (has_key != -1)
                PyErr_SetString(PyExc_ValueError, "Circular reference detected");
            goto bail;
        }
        if (PyDict_SetItem(s->markers, ident, seq)) {
            goto bail;
        }
    }

    iter = PyObject_GetIter(seq);
    if (iter == NULL)
        goto bail;

    if (JSON_Accu_Accumulate(rval, open_array))
        goto bail;
    if (s->indent != Py_None) {
        /* TODO: DOES NOT RUN */
        indent_level += 1;
        /*
            newline_indent = '\n' + (_indent * _current_indent_level)
            separator = _item_separator + newline_indent
            buf += newline_indent
        */
    }
    while ((obj = PyIter_Next(iter))) {
        if (i) {
            if (JSON_Accu_Accumulate(rval, s->item_separator))
                goto bail;
        }
        if (encoder_listencode_obj(s, rval, obj, indent_level))
            goto bail;
        i++;
        Py_CLEAR(obj);
    }
    Py_CLEAR(iter);
    if (PyErr_Occurred())
        goto bail;
    if (ident != NULL) {
        if (PyDict_DelItem(s->markers, ident))
            goto bail;
        Py_CLEAR(ident);
    }
    if (s->indent != Py_None) {
        /* TODO: DOES NOT RUN */
        indent_level -= 1;
        /*
            yield '\n' + (_indent * _current_indent_level)
        */
    }
    if (JSON_Accu_Accumulate(rval, close_array))
        goto bail;
    return 0;

bail:
    Py_XDECREF(obj);
    Py_XDECREF(iter);
    Py_XDECREF(ident);
    return -1;
}

static void
encoder_dealloc(PyObject *self)
{
    /* bpo-31095: UnTrack is needed before calling any callbacks */
    PyObject_GC_UnTrack(self);
    encoder_clear(self);
    Py_TYPE(self)->tp_free(self);
}

static int
encoder_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyEncoderObject *s;
    assert(PyEncoder_Check(self));
    s = (PyEncoderObject *)self;
    Py_VISIT(s->markers);
    Py_VISIT(s->defaultfn);
    Py_VISIT(s->encoder);
    Py_VISIT(s->encoding);
    Py_VISIT(s->indent);
    Py_VISIT(s->key_separator);
    Py_VISIT(s->item_separator);
    Py_VISIT(s->key_memo);
    Py_VISIT(s->sort_keys);
    Py_VISIT(s->item_sort_kw);
    Py_VISIT(s->item_sort_key);
    Py_VISIT(s->max_long_size);
    Py_VISIT(s->min_long_size);
    Py_VISIT(s->Decimal);
    return 0;
}

static int
encoder_clear(PyObject *self)
{
    /* Deallocate Encoder */
    PyEncoderObject *s;
    assert(PyEncoder_Check(self));
    s = (PyEncoderObject *)self;
    Py_CLEAR(s->markers);
    Py_CLEAR(s->defaultfn);
    Py_CLEAR(s->encoder);
    Py_CLEAR(s->encoding);
    Py_CLEAR(s->indent);
    Py_CLEAR(s->key_separator);
    Py_CLEAR(s->item_separator);
    Py_CLEAR(s->key_memo);
    Py_CLEAR(s->skipkeys_bool);
    Py_CLEAR(s->sort_keys);
    Py_CLEAR(s->item_sort_kw);
    Py_CLEAR(s->item_sort_key);
    Py_CLEAR(s->max_long_size);
    Py_CLEAR(s->min_long_size);
    Py_CLEAR(s->Decimal);
    return 0;
}

PyDoc_STRVAR(encoder_doc, "_iterencode(obj, _current_indent_level) -> iterable");

static
PyTypeObject PyEncoderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "simplejson._speedups.Encoder",       /* tp_name */
    sizeof(PyEncoderObject), /* tp_basicsize */
    0,                    /* tp_itemsize */
    encoder_dealloc, /* tp_dealloc */
    0,                    /* tp_print */
    0,                    /* tp_getattr */
    0,                    /* tp_setattr */
    0,                    /* tp_compare */
    0,                    /* tp_repr */
    0,                    /* tp_as_number */
    0,                    /* tp_as_sequence */
    0,                    /* tp_as_mapping */
    0,                    /* tp_hash */
    encoder_call,         /* tp_call */
    0,                    /* tp_str */
    0,                    /* tp_getattro */
    0,                    /* tp_setattro */
    0,                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,   /* tp_flags */
    encoder_doc,          /* tp_doc */
    encoder_traverse,     /* tp_traverse */
    encoder_clear,        /* tp_clear */
    0,                    /* tp_richcompare */
    0,                    /* tp_weaklistoffset */
    0,                    /* tp_iter */
    0,                    /* tp_iternext */
    0,                    /* tp_methods */
    encoder_members,      /* tp_members */
    0,                    /* tp_getset */
    0,                    /* tp_base */
    0,                    /* tp_dict */
    0,                    /* tp_descr_get */
    0,                    /* tp_descr_set */
    0,                    /* tp_dictoffset */
    0,                    /* tp_init */
    0,                    /* tp_alloc */
    encoder_new,          /* tp_new */
    0,                    /* tp_free */
};

static PyMethodDef speedups_methods[] = {
    {"encode_basestring_ascii",
        (PyCFunction)py_encode_basestring_ascii,
        METH_O,
        pydoc_encode_basestring_ascii},
    {"scanstring",
        (PyCFunction)py_scanstring,
        METH_VARARGS,
        pydoc_scanstring},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(module_doc,
"simplejson speedups\n");

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_speedups",        /* m_name */
    module_doc,         /* m_doc */
    -1,                 /* m_size */
    speedups_methods,   /* m_methods */
    NULL,               /* m_reload */
    NULL,               /* m_traverse */
    NULL,               /* m_clear*/
    NULL,               /* m_free */
};
#endif

PyObject *
import_dependency(char *module_name, char *attr_name)
{
    PyObject *rval;
    PyObject *module = PyImport_ImportModule(module_name);
    if (module == NULL)
        return NULL;
    rval = PyObject_GetAttrString(module, attr_name);
    Py_DECREF(module);
    return rval;
}

static int
init_constants(void)
{
    JSON_NaN = JSON_InternFromString("NaN");
    if (JSON_NaN == NULL)
        return 0;
    JSON_Infinity = JSON_InternFromString("Infinity");
    if (JSON_Infinity == NULL)
        return 0;
    JSON_NegInfinity = JSON_InternFromString("-Infinity");
    if (JSON_NegInfinity == NULL)
        return 0;
#if PY_MAJOR_VERSION >= 3
    JSON_EmptyUnicode = PyUnicode_New(0, 127);
#else /* PY_MAJOR_VERSION >= 3 */
    JSON_EmptyStr = PyString_FromString("");
    if (JSON_EmptyStr == NULL)
        return 0;
    JSON_EmptyUnicode = PyUnicode_FromUnicode(NULL, 0);
#endif /* PY_MAJOR_VERSION >= 3 */
    if (JSON_EmptyUnicode == NULL)
        return 0;

    return 1;
}

static PyObject *
moduleinit(void)
{
    PyObject *m;
    if (PyType_Ready(&PyScannerType) < 0)
        return NULL;
    if (PyType_Ready(&PyEncoderType) < 0)
        return NULL;
    if (!init_constants())
        return NULL;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_speedups", speedups_methods, module_doc);
#endif
    Py_INCREF((PyObject*)&PyScannerType);
    PyModule_AddObject(m, "make_scanner", (PyObject*)&PyScannerType);
    Py_INCREF((PyObject*)&PyEncoderType);
    PyModule_AddObject(m, "make_encoder", (PyObject*)&PyEncoderType);
    RawJSONType = import_dependency("simplejson.raw_json", "RawJSON");
    if (RawJSONType == NULL)
        return NULL;
    JSONDecodeError = import_dependency("simplejson.errors", "JSONDecodeError");
    if (JSONDecodeError == NULL)
        return NULL;
    return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit__speedups(void)
{
    return moduleinit();
}
#else
void
init_speedups(void)
{
    moduleinit();
}
#endif
