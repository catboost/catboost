"""Get Unicode argv strings in Python 2 on Windows

get_full_unicode_argv based on
http://code.activestate.com/recipes/572200/

argv_setter_hook based on
https://mail.python.org/pipermail/python-list/2016-June/710183.html
"""

import sys
from ctypes import WinDLL, c_int, POINTER, byref
from ctypes.wintypes import LPCWSTR, LPWSTR

kernel32 = WinDLL("kernel32", use_last_error=True)
shell32 = WinDLL("shell32", use_last_error=True)

GetCommandLineW = kernel32.GetCommandLineW
GetCommandLineW.argtypes = ()
GetCommandLineW.restype = LPCWSTR

CommandLineToArgvW = shell32.CommandLineToArgvW
CommandLineToArgvW.argtypes = (LPCWSTR, POINTER(c_int))
CommandLineToArgvW.restype = POINTER(LPWSTR)

LocalFree = kernel32.LocalFree


def get_full_unicode_argv():
	cmd = GetCommandLineW()
	argc = c_int(0)
	argv = CommandLineToArgvW(cmd, byref(argc))
	py_argv = [arg for i, arg in zip(range(argc.value), argv)]
	LocalFree(argv)
	return py_argv

def get_unicode_argv():
	if original_argv == [""]:
		return [u""]
	
	new_argv = get_full_unicode_argv()[-len(original_argv):]
	
	if original_argv[0] == "-c":
		new_argv[0] = u"-c"
	
	return new_argv


original_argv = None

def argv_setter_hook(path):
	global original_argv
	
	if original_argv is not None: # already got it
		raise ImportError
	
	try:
		original_argv = sys.argv
	except AttributeError:
		pass
	else:
		enable()
	finally:
		raise ImportError

def enable():
	global original_argv
	
	if original_argv is None:
		try:
			original_argv = sys.argv
		except AttributeError: # in sitecustomize in Python 2
			sys.path_hooks[:0] = [argv_setter_hook]
			return
	
	sys.argv = get_unicode_argv()

def disable():
	if original_argv is not None:
		sys.argv = original_argv
