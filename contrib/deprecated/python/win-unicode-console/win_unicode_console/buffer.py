
import ctypes
from ctypes import (byref, POINTER, pythonapi, 
	c_int, c_char, c_char_p, c_void_p, py_object, c_ssize_t)

from .info import PY2


c_ssize_p = POINTER(c_ssize_t)

PyObject_GetBuffer = pythonapi.PyObject_GetBuffer
PyBuffer_Release = pythonapi.PyBuffer_Release


PyBUF_SIMPLE = 0
PyBUF_WRITABLE = 1


class Py_buffer(ctypes.Structure):
	_fields_ = [
		("buf", c_void_p),
		("obj", py_object),
		("len", c_ssize_t),
		("itemsize", c_ssize_t),
		("readonly", c_int),
		("ndim", c_int),
		("format", c_char_p),
		("shape", c_ssize_p),
		("strides", c_ssize_p),
		("suboffsets", c_ssize_p),
		("internal", c_void_p)
	]
	
	if PY2:
		_fields_.insert(-1, ("smalltable", c_ssize_t * 2))
	
	@classmethod
	def get_from(cls, obj, flags=PyBUF_SIMPLE):
		buf = cls()
		PyObject_GetBuffer(py_object(obj), byref(buf), flags)
		return buf
	
	def release(self):
		PyBuffer_Release(byref(self))


def get_buffer(obj, writable=False):
	buf = Py_buffer.get_from(obj, PyBUF_WRITABLE if writable else PyBUF_SIMPLE)
	try:
		buffer_type = c_char * buf.len
		return buffer_type.from_address(buf.buf)
	finally:
		buf.release()

