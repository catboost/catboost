
import sys
import platform


WINDOWS = platform.system().lower() == "windows"
PY2 = sys.version_info.major < 3

def check_Windows():
	current_platform = platform.system()
	
	if not WINDOWS:
		raise RuntimeError("available only for Windows, not {}.".format(current_platform))

def check_PY2():
	if not PY2:
		raise RuntimeError("needed only in Python 2")
