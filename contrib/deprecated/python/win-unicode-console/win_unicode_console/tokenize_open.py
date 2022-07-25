"""Backport of tokenize.open from Python 3.5

This is the exact Python 3.5 with the following differences:
 - detect_encoding_ex is detect_encoding from Python 3.5 returning also a bool whether a cookie was found
 - detect_encoding calls detect_encoding_ex, so that its signature is the same as in Python 3.5
 - function read_source_lines was added
"""

from codecs import lookup, BOM_UTF8
from io import TextIOWrapper, open as _builtin_open
import re

re_ASCII = 256 # not present in Python 2
cookie_re = re.compile(r'^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)', re_ASCII)
blank_re = re.compile(br'^[ \t\f]*(?:[#\r\n]|$)', re_ASCII)


def _get_normal_name(orig_enc):
	"""Imitates get_normal_name in tokenizer.c."""
	# Only care about the first 12 characters.
	enc = orig_enc[:12].lower().replace("_", "-")
	if enc == "utf-8" or enc.startswith("utf-8-"):
		return "utf-8"
	if enc in ("latin-1", "iso-8859-1", "iso-latin-1") or \
			enc.startswith(("latin-1-", "iso-8859-1-", "iso-latin-1-")):
		return "iso-8859-1"
	return orig_enc


def detect_encoding(readline):
	"""
	The detect_encoding() function is used to detect the encoding that should
	be used to decode a Python source file.  It requires one argument, readline,
	in the same way as the tokenize() generator.
	
	It will call readline a maximum of twice, and return the encoding used
	(as a string) and a list of any lines (left as bytes) it has read in.
	
	It detects the encoding from the presence of a utf-8 bom or an encoding
	cookie as specified in pep-0263.  If both a bom and a cookie are present,
	but disagree, a SyntaxError will be raised.  If the encoding cookie is an
	invalid charset, raise a SyntaxError.  Note that if a utf-8 bom is found,
	'utf-8-sig' is returned.
	
	If no encoding is specified, then the default of 'utf-8' will be returned.
	"""
	
	return detect_encoding_ex(readline)[:2]


def detect_encoding_ex(readline):
	try:
		filename = readline.__self__.name
	except AttributeError:
		filename = None
	bom_found = False
	encoding = None
	default = 'utf-8'
	def read_or_stop():
		try:
			return readline()
		except StopIteration:
			return b''
	
	def find_cookie(line):
		try:
			# Decode as UTF-8. Either the line is an encoding declaration,
			# in which case it should be pure ASCII, or it must be UTF-8
			# per default encoding.
			line_string = line.decode('utf-8')
		except UnicodeDecodeError:
			msg = "invalid or missing encoding declaration"
			if filename is not None:
				msg = '{} for {!r}'.format(msg, filename)
			raise SyntaxError(msg)
		
		match = cookie_re.match(line_string)
		if not match:
			return None
		encoding = _get_normal_name(match.group(1))
		try:
			codec = lookup(encoding)
		except LookupError:
			# This behaviour mimics the Python interpreter
			if filename is None:
				msg = "unknown encoding: " + encoding
			else:
				msg = "unknown encoding for {!r}: {}".format(filename,
						encoding)
			raise SyntaxError(msg)
		
		if bom_found:
			if encoding != 'utf-8':
				# This behaviour mimics the Python interpreter
				if filename is None:
					msg = 'encoding problem: utf-8'
				else:
					msg = 'encoding problem for {!r}: utf-8'.format(filename)
				raise SyntaxError(msg)
			encoding += '-sig'
		return encoding
	
	first = read_or_stop()
	if first.startswith(BOM_UTF8):
		bom_found = True
		first = first[3:]
		default = 'utf-8-sig'
	if not first:
		return default, [], False
	
	encoding = find_cookie(first)
	if encoding:
		return encoding, [first], True
	if not blank_re.match(first):
		return default, [first], False
	
	second = read_or_stop()
	if not second:
		return default, [first], False
	
	encoding = find_cookie(second)
	if encoding:
		return encoding, [first, second], True
	
	return default, [first, second], False


def open(filename):
	"""Open a file in read only mode using the encoding detected by
	detect_encoding().
	"""
	buffer = _builtin_open(filename, 'rb')
	try:
		encoding, lines = detect_encoding(buffer.readline)
		buffer.seek(0)
		text = TextIOWrapper(buffer, encoding, line_buffering=True)
		text.mode = 'r'
		return text
	except:
		buffer.close()
		raise

def read_source_lines(filename):
	buffer = _builtin_open(filename, 'rb')
	try:
		encoding, lines, cookie_present = detect_encoding_ex(buffer.readline)
		buffer.seek(0)
		text = TextIOWrapper(buffer, encoding, line_buffering=True)
		text.mode = 'r'
	except:
		buffer.close()
		raise
	
	with text:
		if cookie_present:
			for i in lines:
				yield text.readline().replace("coding", "Coding")
				# so compile() won't complain about encoding declatation in a Unicode string
				# see 2.7/Python/ast.c:228
		
		for line in text:
			yield line
