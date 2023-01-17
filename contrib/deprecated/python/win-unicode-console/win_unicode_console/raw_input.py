
from .info import check_PY2
check_PY2()

import __builtin__ as builtins
import sys
from ctypes import pythonapi, c_char_p, c_void_p, py_object

from .streams import STDIN, STDOUT
from .readline_hook import check_encodings, stdio_readline


original_raw_input = builtins.raw_input
original_input = builtins.input

RETURN_UNICODE = True


PyOS_Readline = pythonapi.PyOS_Readline
PyOS_Readline.restype = c_char_p
PyOS_Readline.argtypes = [c_void_p, c_void_p, c_char_p]

PyFile_AsFile = pythonapi.PyFile_AsFile
PyFile_AsFile.restype = c_void_p
PyFile_AsFile.argtypes = [py_object]

STDIN_FILE_POINTER = PyFile_AsFile(sys.stdin)
STDOUT_FILE_POINTER = PyFile_AsFile(sys.stdout)


def stdout_encode(s):
	if isinstance(s, bytes):
		return s
	encoding = sys.stdout.encoding
	errors = sys.stdout.errors
	if errors is not None:
		return s.encode(encoding, errors)
	else:
		return s.encode(encoding)

def stdin_encode(s):
	if isinstance(s, bytes):
		return s
	encoding = sys.stdin.encoding
	errors = sys.stdin.errors
	if errors is not None:
		return s.encode(encoding, errors)
	else:
		return s.encode(encoding)

def stdin_decode(b):
	if isinstance(b, unicode):
		return b
	encoding = sys.stdin.encoding
	errors = sys.stdin.errors
	if errors is not None:
		return b.decode(encoding, errors)
	else:
		return b.decode(encoding)

def readline(prompt=""):
	check_encodings()
	prompt_bytes = stdout_encode(prompt)
	line_bytes = PyOS_Readline(STDIN_FILE_POINTER, STDOUT_FILE_POINTER, prompt_bytes)
	if line_bytes is None:
		raise KeyboardInterrupt
	else:
		return line_bytes


def raw_input(prompt=""):
	"""raw_input([prompt]) -> string

Read a string from standard input.  The trailing newline is stripped.
If the user hits EOF (Unix: Ctl-D, Windows: Ctl-Z+Return), raise EOFError.
On Unix, GNU readline is used if enabled.  The prompt string, if given,
is printed without a trailing newline before reading."""
	
	sys.stderr.flush()
	
	tty = STDIN.is_a_TTY() and STDOUT.is_a_TTY()
	
	if RETURN_UNICODE:
		if tty:
			line_bytes = readline(prompt)
			line = stdin_decode(line_bytes)
		else:
			line = stdio_readline(prompt)
		
	else:
		if tty:
			line = readline(prompt)
		else:
			line_unicode = stdio_readline(prompt)
			line = stdin_encode(line_unicode)
	
	if line:
		return line[:-1] # strip strailing "\n"
	else:
		raise EOFError

def input(prompt=""):
	"""input([prompt]) -> value

Equivalent to eval(raw_input(prompt))."""
	
	string = stdin_decode(raw_input(prompt))
	
	caller_frame = sys._getframe(1)
	globals = caller_frame.f_globals
	locals = caller_frame.f_locals
	
	return eval(string, globals, locals)


def enable(return_unicode=RETURN_UNICODE):
	global RETURN_UNICODE
	RETURN_UNICODE = return_unicode
	
	builtins.raw_input = raw_input
	builtins.input = input

def disable():
	builtins.raw_input = original_raw_input
	builtins.input = original_input
