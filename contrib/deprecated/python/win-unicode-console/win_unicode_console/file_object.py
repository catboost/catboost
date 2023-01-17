
from .info import check_PY2
check_PY2()

import ctypes
from ctypes import (byref, pythonapi, 
	c_int, c_char_p, c_void_p, py_object, c_ssize_t)


class FileObject(ctypes.Structure):
	_fields_ = [
		#("_ob_next", c_void_p),
		#("_ob_prev", c_void_p),
		("ob_refcnt", c_ssize_t),
		("ob_type", c_void_p),
		
		("fp", c_void_p),
		("name", py_object),
		("mode", py_object),
		("close", c_void_p),
		("softspace", c_int),
		("binary", c_int),
		("buf", c_char_p),
		("bufend", c_char_p),
		("bufptr", c_char_p),
		("setbuf", c_char_p),
		("univ_newline", c_int),
		("newlinetypes", c_int),
		("skipnextlf", c_int),
		("encoding", py_object),
		("errors", py_object),
		("weakreflist", py_object),
		("unlocked_count", c_int),
		("readable", c_int),
		("writable", c_int),
	]
	
	@classmethod
	def from_file(cls, f):
		if not isinstance(f, file):
			raise TypeError("f has to be a file")
		
		return cls.from_address(id(f))
	
	def set_encoding(self, encoding):
		if not isinstance(encoding, str):
			raise TypeError("encoding has to be a str")
		
		pythonapi.PyFile_SetEncoding(byref(self), encoding)
	
	def copy_file_pointer(self, f):
		if not isinstance(f, file):
			raise TypeError("f has to be a file")
		
		self.fp = pythonapi.PyFile_AsFile(py_object(f))
