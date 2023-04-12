
from __future__ import print_function # PY2

import __main__
import argparse
import sys
import traceback
import tokenize
from ctypes import pythonapi, POINTER, c_long, cast
from types import CodeType as Code

from . import console, enable, disable
from .info import PY2


inspect_flag = cast(pythonapi.Py_InspectFlag, POINTER(c_long)).contents

def set_inspect_flag(value):
	inspect_flag.value = int(value)


CODE_FIELDS = ["argcount", "kwonlyargcount", "nlocals", "stacksize", 
		"flags", "code", "consts", "names", "varnames", "filename", 
		"name", 	"firstlineno", "lnotab", "freevars", "cellvars"]
if PY2:
	CODE_FIELDS.remove("kwonlyargcount")

def update_code(codeobj, **kwargs):
	def field_values():
		for field in CODE_FIELDS:
			original_value = getattr(codeobj, "co_{}".format(field))
			value = kwargs.get(field, original_value)
			yield value
	
	return Code(*field_values())

def update_code_recursively(codeobj, **kwargs):
	updated = {}
	
	def update(codeobj, **kwargs):
		result = updated.get(codeobj, None)
		if result is not None:
			return result
		
		if any(isinstance(c, Code) for c in codeobj.co_consts):
			consts = tuple(update(c, **kwargs) if isinstance(c, Code) else c
				for c in codeobj.co_consts)
		else:
			consts = codeobj.co_consts
		
		result = update_code(codeobj, consts=consts, **kwargs)
		updated[codeobj] = result
		return result
	
	return update(codeobj, **kwargs)


def get_code(path):
	if PY2:
		from .tokenize_open import read_source_lines
		source = u"".join(read_source_lines(path))
	else:
		with tokenize.open(path) as f: # opens with detected source encoding
			source = f.read()
	
	try:
		code = compile(source, path, "exec", dont_inherit=True)
	except UnicodeEncodeError:
		code = compile(source, "<encoding error>", "exec", dont_inherit=True)
		if PY2:
			path = path.encode("utf-8")
		code = update_code_recursively(code, filename=path)
			# so code constains correct filename (even if it contains Unicode)
			# and tracebacks show contents of code lines
	
	return code


def print_exception_without_first_line(etype, value, tb, limit=None, file=None, chain=True):
	if file is None:
		file = sys.stderr
	
	lines = iter(traceback.TracebackException(
		type(value), value, tb, limit=limit).format(chain=chain))
	
	next(lines)
	for line in lines:
		print(line, file=file, end="")


def run_script(args):
	sys.argv = [args.script] + args.script_arguments
	path = args.script
	__main__.__file__ = path
	
	try:
		code = get_code(path)
	except Exception as e:
		traceback.print_exception(e.__class__, e, None, file=sys.stderr)
	else:
		try:
			exec(code, __main__.__dict__)
		except BaseException as e:
			if not sys.flags.inspect and isinstance(e, SystemExit):
				raise
				
			elif PY2: # Python 2 produces tracebacks in mixed encoding (!)
				etype, e, tb = sys.exc_info()
				for line in traceback.format_exception(etype, e, tb.tb_next):
					line = line.decode("utf-8", "replace")
					try:
						sys.stderr.write(line)
					except UnicodeEncodeError:
						line = line.encode(sys.stderr.encoding, "backslashreplace")
						sys.stderr.write(line)
					
					sys.stderr.flush() # is this needed?
				
			else: # PY3
				traceback.print_exception(e.__class__, e, e.__traceback__.tb_next, file=sys.stderr)

def run_init(args):
	if args.init == "enable":
		enable()
	elif args.init == "disable":
		disable()
	elif args.init == "module":
		__import__(args.module)
	elif args.init == "none":
		pass
	else:
		raise ValueError("unknown runner init mode {}".format(repr(args.init)))

def run_with_custom_repl(args):
	run_init(args)
	
	if args.script:
		run_script(args)
	
	if sys.flags.interactive or not args.script:
		if sys.flags.interactive and not args.script:
			console.print_banner()
		try:
			console.enable()
		finally:
			set_inspect_flag(0)

def run_with_standard_repl(args):
	run_init(args)
	
	if args.script:
		run_script(args)
	
	if sys.flags.interactive and not args.script:
		console.print_banner()

def run_arguments():
	parser = argparse.ArgumentParser(description="Runs a script after customizable initialization. By default, win_unicode_console is enabled.")
	
	init_group = parser.add_mutually_exclusive_group()
	init_group.add_argument(
		"-e", "--init-enable", dest="init", action="store_const", const="enable", 
		help="enable win_unicode_console on init (default)")
	init_group.add_argument(
		"-d", "--init-disable", dest="init", action="store_const", const="disable", 
		help="disable win_unicode_console on init")
	init_group.add_argument(
		"-m", "--init-module", dest="module", 
		help="import the given module on init")
	init_group.add_argument(
		"-n", "--no-init", dest="init", action="store_const", const="none", 
		help="do nothing special on init")
	parser.set_defaults(init="enable")
	
	repl_group = parser.add_mutually_exclusive_group()
	repl_group.add_argument(
		"-s", "--standard-repl", dest="use_repl", action="store_false", 
		help="use the standard Python REPL (default)")
	repl_group.add_argument(
		"-c", "--custom-repl", dest="use_repl", action="store_true", 
		help="use win_unicode_console.console REPL")
	parser.set_defaults(use_repl=False)
	
	parser.add_argument("script", nargs="?")
	parser.add_argument("script_arguments", nargs=argparse.REMAINDER, metavar="script-arguments")
	
	try:
		args = parser.parse_args(sys.argv[1:])
	except SystemExit:
		set_inspect_flag(0)	# don't go interactive after printing help
		raise
	
	if args.module:
		args.init = "module"
	
	if args.use_repl:
		run_with_custom_repl(args)
	else:
		run_with_standard_repl(args)
