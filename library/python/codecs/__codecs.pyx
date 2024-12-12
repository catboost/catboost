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
        void Encode(TStringBuf data, TString& res) nogil except +
        void Decode(TStringBuf data, TString& res) nogil except +

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


def get_codec_id(name):
    if name == "lz4":
        return 6051
    elif name == "snappy":
        return 50986
    elif name == "std08_1":
        return 55019
    elif name == "std08_3":
        return 23308
    elif name == "std08_7":
        return 33533
    elif name == "brotli_1":
        return 48947
    elif name == "brotli_10":
        return 43475
    elif name == "brotli_11":
        return 7241
    elif name == "brotli_2":
        return 63895
    elif name == "brotli_3":
        return 11408
    elif name == "brotli_4":
        return 47136
    elif name == "brotli_5":
        return 45284
    elif name == "brotli_6":
        return 63219
    elif name == "brotli_7":
        return 59675
    elif name == "brotli_8":
        return 40233
    elif name == "brotli_9":
        return 10380
    else:
        raise RuntimeError("Unknown code name: " + name)


def list_all_codecs():
    cdef TString res = ListAllCodecsAsString()

    return from_bytes(res.c_str()[:res.length()]).split(',')
