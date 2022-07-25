
import io
import sys
import time
from ctypes import byref, c_ulong

from .buffer import get_buffer
from .info import WINDOWS, PY2

if PY2:
	from .file_object import FileObject


if WINDOWS:
	from ctypes import WinDLL, get_last_error, set_last_error, WinError
	from msvcrt import get_osfhandle
	
	kernel32 = WinDLL("kernel32", use_last_error=True)
	ReadConsoleW = kernel32.ReadConsoleW
	WriteConsoleW = kernel32.WriteConsoleW
	GetConsoleMode = kernel32.GetConsoleMode


ERROR_SUCCESS = 0
ERROR_INVALID_HANDLE = 6
ERROR_NOT_ENOUGH_MEMORY = 8
ERROR_OPERATION_ABORTED = 995

EOF = b"\x1a"

MAX_BYTES_WRITTEN = 32767	# arbitrary because WriteConsoleW ability to write big buffers depends on heap usage


class StandardStreamInfo:
	def __init__(self, name, standard_fileno):
		self.name = name
		self.fileno = standard_fileno
		self.handle = get_osfhandle(standard_fileno) if WINDOWS else None
	
	def __repr__(self):
		return "<{} '{}' fileno={} handle={}>".format(self.__class__.__name__, self.name, self.fileno, self.handle)
	
	@property
	def stream(self):
		return getattr(sys, self.name)
	
	def is_a_TTY(self):
		# the test used in input()
		try:
			get_fileno = self.stream.fileno
		except AttributeError: # e.g. StringIO in Python 2
			return False
		
		try:
			fileno = get_fileno()
		except io.UnsupportedOperation:
			return False
		else:
			return fileno == self.fileno and self.stream.isatty()
	
	def is_a_console(self):
		if self.handle is None:
			return False
		
		if GetConsoleMode(self.handle, byref(c_ulong())):
			return True
		else:
			last_error = get_last_error()
			if last_error == ERROR_INVALID_HANDLE:
				return False
			else:
				raise WinError(last_error)
	
	def should_be_fixed(self):
		if self.stream is None:	# e.g. with IDLE
			return True
		
		return self.is_a_TTY() and self.is_a_console()

STDIN = StandardStreamInfo("stdin", standard_fileno=0)
STDOUT = StandardStreamInfo("stdout", standard_fileno=1)
STDERR = StandardStreamInfo("stderr", standard_fileno=2)


class _ReprMixin:
	def __repr__(self):
		modname = self.__class__.__module__
		
		if PY2:
			clsname = self.__class__.__name__
		else:
			clsname = self.__class__.__qualname__
		
		attributes = []
		for name in ["name", "encoding"]:
			try:
				value = getattr(self, name)
			except AttributeError:
				pass
			else:
				attributes.append("{}={}".format(name, repr(value)))
		
		return "<{}.{} {}>".format(modname, clsname, " ".join(attributes))


class WindowsConsoleRawIOBase(_ReprMixin, io.RawIOBase):
	def __init__(self, name, handle, fileno):
		self.name = name
		self.handle = handle
		self.file_no = fileno
	
	def fileno(self):
		return self.file_no
	
	def isatty(self):
		# PY3 # super().isatty()	# for close check in default implementation
		super(WindowsConsoleRawIOBase, self).isatty()
		return True

class WindowsConsoleRawReader(WindowsConsoleRawIOBase):
	def readable(self):
		return True
	
	def readinto(self, b):
		bytes_to_be_read = len(b)
		if not bytes_to_be_read:
			return 0
		elif bytes_to_be_read % 2:
			raise ValueError("cannot read odd number of bytes from UTF-16-LE encoded console")
		
		buffer = get_buffer(b, writable=True)
		code_units_to_be_read = bytes_to_be_read // 2
		code_units_read = c_ulong()
		
		set_last_error(ERROR_SUCCESS)
		ReadConsoleW(self.handle, buffer, code_units_to_be_read, byref(code_units_read), None)
		last_error = get_last_error()
		if last_error == ERROR_OPERATION_ABORTED:
			time.sleep(0.1)	# wait for KeyboardInterrupt
		if last_error != ERROR_SUCCESS:
			raise WinError(last_error)
		
		if buffer[0] == EOF:
			return 0
		else:
			return 2 * code_units_read.value # bytes read

class WindowsConsoleRawWriter(WindowsConsoleRawIOBase):
	def writable(self):
		return True
	
	def write(self, b):
		bytes_to_be_written = len(b)
		buffer = get_buffer(b)
		code_units_to_be_written = min(bytes_to_be_written, MAX_BYTES_WRITTEN) // 2
		code_units_written = c_ulong()
		
		if code_units_to_be_written == 0 != bytes_to_be_written:
			raise ValueError("two-byte code units expected, just one byte given")
		
		if not WriteConsoleW(self.handle, buffer, code_units_to_be_written, byref(code_units_written), None):
			exc = WinError(get_last_error())
			if exc.winerror == ERROR_NOT_ENOUGH_MEMORY:
				exc.strerror += " Try to lower `win_unicode_console.streams.MAX_BYTES_WRITTEN`."
			raise exc
		
		return 2 * code_units_written.value # bytes written


class _TextStreamWrapperMixin(_ReprMixin):
	def __init__(self, base):
		self.base = base
	
	@property
	def encoding(self):
		return self.base.encoding
	
	@property
	def errors(self):
		return self.base.errors
	
	@property
	def line_buffering(self):
		return self.base.line_buffering
	
	def seekable(self):
		return self.base.seekable()
	
	def readable(self):
		return self.base.readable()
	
	def writable(self):
		return self.base.writable()
	
	def flush(self):
		self.base.flush()
	
	def close(self):
		self.base.close()
	
	@property
	def closed(self):
		return self.base.closed
	
	@property
	def name(self):
		return self.base.name
	
	def fileno(self):
		return self.base.fileno()
	
	def isatty(self):
		return self.base.isatty()
	
	def write(self, s):
		return self.base.write(s)
	
	def tell(self):
		return self.base.tell()
	
	def truncate(self, pos=None):
		return self.base.truncate(pos)
	
	def seek(self, cookie, whence=0):
		return self.base.seek(cookie, whence)
	
	def read(self, size=None):
		return self.base.read(size)
	
	def __next__(self):
		return next(self.base)
	
	def readline(self, size=-1):
		return self.base.readline(size)
	
	@property
	def newlines(self):
		return self.base.newlines

class TextStreamWrapper(_TextStreamWrapperMixin, io.TextIOBase):
	pass

class TextTranscodingWrapper(TextStreamWrapper):
	encoding = None # disable the descriptor
	
	def __init__(self, base, encoding):
		# PY3 # super().__init__(base)
		super(TextTranscodingWrapper, self).__init__(base)
		self.encoding = encoding

class StrStreamWrapper(TextStreamWrapper):
	def write(self, s):
		if isinstance(s, bytes):
			s = s.decode(self.encoding)
		
		self.base.write(s)

if PY2:
	class FileobjWrapper(_TextStreamWrapperMixin, file):
		def __init__(self, base, f):
			super(FileobjWrapper, self).__init__(base)
			fileobj = self._fileobj = FileObject.from_file(self)
			fileobj.set_encoding(base.encoding)
			fileobj.copy_file_pointer(f)
			fileobj.readable = base.readable()
			fileobj.writable = base.writable()
		
		# needed for the right interpretation of unicode literals in interactive mode when win_unicode_console is enabled in sitecustomize since Py_Initialize changes encoding afterwards
		def _reset_encoding(self):
			self._fileobj.set_encoding(self.base.encoding)
		
		def readline(self, size=-1):
			self._reset_encoding()
			return self.base.readline(size)


if WINDOWS:
	stdin_raw = WindowsConsoleRawReader("<stdin>", STDIN.handle, STDIN.fileno)
	stdout_raw = WindowsConsoleRawWriter("<stdout>", STDOUT.handle, STDOUT.fileno)
	stderr_raw = WindowsConsoleRawWriter("<stderr>", STDERR.handle, STDERR.fileno)
	
	stdin_text = io.TextIOWrapper(io.BufferedReader(stdin_raw), encoding="utf-16-le", line_buffering=True)
	stdout_text = io.TextIOWrapper(io.BufferedWriter(stdout_raw), encoding="utf-16-le", line_buffering=True)
	stderr_text = io.TextIOWrapper(io.BufferedWriter(stderr_raw), encoding="utf-16-le", line_buffering=True)
	
	stdin_text_transcoded = TextTranscodingWrapper(stdin_text, encoding="utf-8")
	stdout_text_transcoded = TextTranscodingWrapper(stdout_text, encoding="utf-8")
	stderr_text_transcoded = TextTranscodingWrapper(stderr_text, encoding="utf-8")
	
	stdout_text_str = StrStreamWrapper(stdout_text_transcoded)
	stderr_text_str = StrStreamWrapper(stderr_text_transcoded)
	if PY2:
		stdin_text_fileobj = FileobjWrapper(stdin_text_transcoded, sys.__stdin__)
		stdout_text_str_fileobj = FileobjWrapper(stdout_text_str, sys.__stdout__)


def disable():
	sys.stdin.flush()
	sys.stdout.flush()
	sys.stderr.flush()
	sys.stdin = sys.__stdin__
	sys.stdout = sys.__stdout__
	sys.stderr = sys.__stderr__

# PY3 # def enable(*, stdin=Ellipsis, stdout=Ellipsis, stderr=Ellipsis):
def enable(stdin=Ellipsis, stdout=Ellipsis, stderr=Ellipsis):
	if not WINDOWS:
		return
	
	# defaults
	if PY2:
		if stdin is Ellipsis:
			stdin = stdin_text_fileobj
		if stdout is Ellipsis:
			stdout = stdout_text_str
		if stderr is Ellipsis:
			stderr = stderr_text_str
	else: # transcoding because Python tokenizer cannot handle UTF-16
		if stdin is Ellipsis:
			stdin = stdin_text_transcoded
		if stdout is Ellipsis:
			stdout = stdout_text_transcoded
		if stderr is Ellipsis:
			stderr = stderr_text_transcoded
	
	if stdin is not None and STDIN.should_be_fixed():
		sys.stdin = stdin
	if stdout is not None and STDOUT.should_be_fixed():
		sys.stdout.flush()
		sys.stdout = stdout
	if stderr is not None and STDERR.should_be_fixed():
		sys.stderr.flush()
		sys.stderr = stderr

# PY3 # def enable_only(*, stdin=None, stdout=None, stderr=None):
def enable_only(stdin=None, stdout=None, stderr=None):
	enable(stdin=stdin, stdout=stdout, stderr=stderr)
