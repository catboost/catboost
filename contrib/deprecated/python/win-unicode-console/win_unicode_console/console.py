
from __future__ import print_function # PY2

import __main__
import code
import sys

from .info import PY2


def print_banner(file=sys.stderr):
	print("Python {} on {}".format(sys.version, sys.platform), file=file)
	print('Type "help", "copyright", "credits" or "license" for more information.', file=file)

# PY3 # class InteractiveConsole(code.InteractiveConsole):
class InteractiveConsole(code.InteractiveConsole, object):
	# code.InteractiveConsole without banner
	# exits on EOF
	# also more robust treating of sys.ps1, sys.ps2
	# prints prompt into stderr rather than stdout
	# flushes sys.stderr and sys.stdout
	
	def __init__(self, locals=None, filename="<stdin>"):
		self.done = False
		# PY3 # super().__init__(locals, filename)
		super(InteractiveConsole, self).__init__(locals, filename)
	
	def raw_input(self, prompt=""):
		sys.stderr.write(prompt)
		if PY2:
			return raw_input()
		else:
			return input()
	
	def runcode(self, code):
		# PY3 # super().runcode(code)
		super(InteractiveConsole, self).runcode(code)
		sys.stderr.flush()
		sys.stdout.flush()
	
	def interact(self):
		#sys.ps1 = "~>> "
		#sys.ps2 = "~.. "
		
		try:
			sys.ps1
		except AttributeError:
			sys.ps1 = ">>> "
		
		try:
			sys.ps2
		except AttributeError:
			sys.ps2 = "... "
		
		more = 0
		while not self.done:
			try:
				if more:
					try:
						prompt = sys.ps2
					except AttributeError:
						prompt = ""
				else:
					try:
						prompt = sys.ps1
					except AttributeError:
						prompt = ""
				
				try:
					line = self.raw_input(prompt)
				except EOFError:
					self.on_EOF()
				else:
					more = self.push(line)
				
			except KeyboardInterrupt:
				self.write("\nKeyboardInterrupt\n")
				self.resetbuffer()
				more = 0
	
	def on_EOF(self):
		self.write("\n")
		# PY3 # raise SystemExit from None
		raise SystemExit


running_console = None

def enable():
	global running_console
	
	if running_console is not None:
		raise RuntimeError("interactive console already running")
	else:
		running_console = InteractiveConsole(__main__.__dict__) 
		running_console.interact() 

def disable():
	global running_console
	
	if running_console is None:
		raise RuntimeError("interactive console is not running")
	else:
		running_console.done = True
		running_console = None

