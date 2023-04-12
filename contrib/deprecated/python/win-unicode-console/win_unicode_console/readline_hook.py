
from __future__ import print_function # PY2

import sys
import traceback
import warnings
import ctypes.util
from ctypes import (pythonapi, cdll, cast, 
	c_char_p, c_void_p, c_size_t, CFUNCTYPE)

from .info import WINDOWS

try:
	import pyreadline
except ImportError:
	pyreadline = None


def get_libc():
	if WINDOWS:
		path = "msvcrt"
	else:
		path = ctypes.util.find_library("c")
		if path is None:
			raise RuntimeError("cannot locate libc")
	
	return cdll[path]

LIBC = get_libc()

PyMem_Malloc = pythonapi.PyMem_Malloc
PyMem_Malloc.restype = c_size_t
PyMem_Malloc.argtypes = [c_size_t]

strncpy = LIBC.strncpy
strncpy.restype = c_char_p
strncpy.argtypes = [c_char_p, c_char_p, c_size_t]

HOOKFUNC = CFUNCTYPE(c_char_p, c_void_p, c_void_p, c_char_p)

#PyOS_ReadlineFunctionPointer = c_void_p.in_dll(pythonapi, "PyOS_ReadlineFunctionPointer")


def new_zero_terminated_string(b):
	p = PyMem_Malloc(len(b) + 1)
	strncpy(cast(p, c_char_p), b, len(b) + 1)
	return p

def check_encodings():
	if sys.stdin.encoding != sys.stdout.encoding:
		# raise RuntimeError("sys.stdin.encoding != sys.stdout.encoding, readline hook doesn't know, which one to use to decode prompt")
		
		warnings.warn("sys.stdin.encoding == {!r}, whereas sys.stdout.encoding == {!r}, readline hook consumer may assume they are the same".format(sys.stdin.encoding, sys.stdout.encoding), 
			RuntimeWarning, stacklevel=3)

def stdio_readline(prompt=""):
	sys.stdout.write(prompt)
	sys.stdout.flush()
	return sys.stdin.readline()


class ReadlineHookManager:
	def __init__(self):
		self.readline_wrapper_ref = HOOKFUNC(self.readline_wrapper)
		self.address = cast(self.readline_wrapper_ref, c_void_p).value
		#self.original_address = PyOS_ReadlineFunctionPointer.value
		self.readline_hook = None
	
	def readline_wrapper(self, stdin, stdout, prompt):
		try:
			try:
				check_encodings()
			except RuntimeError:
				traceback.print_exc(file=sys.stderr)
				try:
					prompt = prompt.decode("utf-8")
				except UnicodeDecodeError:
					prompt = ""
				
			else:
				prompt = prompt.decode(sys.stdout.encoding)
			
			try:
				line = self.readline_hook(prompt)
			except KeyboardInterrupt:
				return 0
			else:
				return new_zero_terminated_string(line.encode(sys.stdin.encoding))
			
		except:
			self.restore_original()
			print("Internal win_unicode_console error, disabling custom readline hook...", file=sys.stderr)
			traceback.print_exc(file=sys.stderr)
			return new_zero_terminated_string(b"\n")
	
	def install_hook(self, hook):
		self.readline_hook = hook
		PyOS_ReadlineFunctionPointer.value = self.address
	
	def restore_original(self):
		self.readline_hook = None
		PyOS_ReadlineFunctionPointer.value = self.original_address


class PyReadlineManager:
	def __init__(self):
		self.original_codepage = pyreadline.unicode_helper.pyreadline_codepage
	
	def set_codepage(self, codepage):
		pyreadline.unicode_helper.pyreadline_codepage = codepage
	
	def restore_original(self):
		self.set_codepage(self.original_codepage)

def pyreadline_is_active():
	if not pyreadline:
		return False
	
	ref = pyreadline.console.console.readline_ref
	if ref is None:
		return False
	
	return cast(ref, c_void_p).value == PyOS_ReadlineFunctionPointer.value


manager = ReadlineHookManager()

if pyreadline:
	pyreadline_manager = PyReadlineManager()


# PY3 # def enable(*, use_pyreadline=True):
def enable(use_pyreadline=True):
	check_encodings()
	
	if use_pyreadline and pyreadline:
		pyreadline_manager.set_codepage(sys.stdin.encoding)
			# pyreadline assumes that encoding of all sys.stdio objects is the same
		if not pyreadline_is_active():
			manager.install_hook(stdio_readline)
		
	else:
		manager.install_hook(stdio_readline)

def disable():
	if pyreadline:
		pyreadline_manager.restore_original()
	else:
		manager.restore_original()
