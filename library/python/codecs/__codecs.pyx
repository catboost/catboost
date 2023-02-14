import six

from libcpp cimport bool

from util.generic.string cimport TString, TStringBuf


def to_bytes(s):
    try:
        return s.encode('utf-8')
    except AttributeError:
        pass

    return s


def from_bytes(s):
    if six.PY3:
        return s.decode('utf-8')

    return s


cdef extern from "library/cpp/blockcodecs/codecs.h" namespace "NBlockCodecs":
    cdef cppclass ICodec:
        void Encode(TStringBuf data, TString& res) nogil
        void Decode(TStringBuf data, TString& res) nogil

    cdef const ICodec* Codec(const TStringBuf& name) except +
    cdef TString ListAllCodecsAsString() except +


def dumps(name, data):
    name = to_bytes(name)

    cdef const ICodec* codec = Codec(TStringBuf(name, len(name)))
    cdef TString res
    cdef TStringBuf cdata = TStringBuf(data, len(data))

    with nogil:
        codec.Encode(cdata, res)

    return res.c_str()[:res.length()]


def loads(name, data):
    name = to_bytes(name)

    cdef const ICodec* codec = Codec(TStringBuf(name, len(name)))
    cdef TString res
    cdef TStringBuf cdata = TStringBuf(data, len(data))

    with nogil:
        codec.Decode(cdata, res)

    return res.c_str()[:res.length()]

def list_all_codecs():
    cdef TString res = ListAllCodecsAsString()

    return from_bytes(res.c_str()[:res.length()]).split(',')
