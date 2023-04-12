
from . import streams, console #, readline_hook
from .info import WINDOWS, PY2

if PY2:
	from . import raw_input

if PY2 and WINDOWS:
	from . import unicode_argv


# PY3 # def enable(*, 
def enable(
		stdin = Ellipsis, 
		stdout = Ellipsis, 
		stderr = Ellipsis, 
		use_readline_hook = False, 
		use_pyreadline = True, 
		use_raw_input = True, # PY2
		raw_input__return_unicode = raw_input.RETURN_UNICODE if PY2 else None, 
		use_unicode_argv = False, # PY2, has some issues
		use_repl = False#, 
	):
	
	if not WINDOWS:
		return
	
	streams.enable(stdin=stdin, stdout=stdout, stderr=stderr)
	
	#if use_readline_hook:
	#	readline_hook.enable(use_pyreadline=use_pyreadline)
	
	if PY2 and use_raw_input:
		raw_input.enable(raw_input__return_unicode)
	
	if PY2 and use_unicode_argv:
		unicode_argv.enable()
	
	if use_repl:
		console.enable()

def disable():
	if not WINDOWS:
		return
	
	if console.running_console is not None:
		console.disable()
	
	if PY2:
		unicode_argv.disable()
		raw_input.disable()
	
	#readline_hook.disable()
	streams.disable()
