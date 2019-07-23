from ctypes import (
    byref, POINTER, c_int, c_char, c_char_p,
    c_void_p, py_object, c_ssize_t, pythonapi, Structure
)

c_ssize_p = POINTER(c_ssize_t)


class Py_buffer(Structure):
    _fields_ = [
        ('buf', c_void_p),
        ('obj', py_object),
        ('len', c_ssize_t),
        ('itemsize', c_ssize_t),
        ('readonly', c_int),
        ('ndim', c_int),
        ('format', c_char_p),
        ('shape', c_ssize_p),
        ('strides', c_ssize_p),
        ('suboffsets', c_ssize_p),
        ('smalltable', c_ssize_t * 2),
        ('internal', c_void_p)
    ]


def get_buffer(obj):
    buf = Py_buffer()
    pythonapi.PyObject_GetBuffer(py_object(obj), byref(buf), 0)
    try:
        buffer_type = c_char * buf.len
        return buffer_type.from_address(buf.buf)
    finally:
        pythonapi.PyBuffer_Release(byref(buf))


def test_buffer():
    assert get_buffer(b'test string')
