'''
This package provides several entry points
    * spec_parser looks for code annotations in the form of formatted comments
    * functions defined in toolchain.py:
       * generate_cxx:    python (str) to c++ code, returns a PythonModule
       * compile_cxxfile: c++ (file) to DLL, returns DLL filename
       * compile_cxxcode: c++ (str) to DLL, returns DLL filename
       * compile_pythrancode: python (str) to so/cpp, returns output filename
       * compile_pythranfile: python (file) to so/cpp, returns output filename
       * import_pythrancode: python (str) to module, returns loaded module
       * import_pythranfile: python (file) to module, returns loaded module
       * test_compile: passthrough compile test, raises CompileError Exception.

Basic scenario is to turn a Python AST into C++ code:
>>> code = "def foo(x): return x * 2"
>>> cxx_generator, error_checker = generate_cxx('my_module', code)
>>> cxx = cxx_generator.generate()

To generate a native module, one need to add type information:
>>> cxx = generate_cxx('my_module', code, {'foo':([int],)})

Eventually, the type information can be translated from a string:
>>> spec = spec_parser('#pythran export foo(int)')
>>> cxx = generate_cxx('my_module', code, spec)

Higher level entry points include:
>>> export = '#pythran export foo(int)\\n'
>>> with open('my_module.py', 'w') as fd:
...     _ = fd.write(export + code)
>>> dll_file = compile_pythranfile("my_module.py")
>>> cpp_file = compile_pythranfile("my_module.py",cpponly=True)
>>> dll_file = compile_pythrancode("my_module", export + code)
>>> dll_file = compile_cxxfile("my_module", cpp_file)

It is possible to directly turn pythran code into an imported module:
>>> code = '#pythran export greet()\\ndef greet(): return "demat"'
>>> greeter = import_pythrancode(code)
>>> greeter.greet()
'demat'

Cleanup
>>> import os, glob
>>> for target in glob.glob('my_module.*'):
...     os.remove(target)

'''

import pythran.log
from pythran.config import get_include
from pythran.toolchain import (generate_cxx, compile_cxxfile, compile_cxxcode,
                               compile_pythrancode, compile_pythranfile,
                               import_pythrancode, import_pythranfile,
                               test_compile)
from pythran.spec import spec_parser
from pythran.spec import load_specfile
from pythran.dist import PythranExtension
from pythran.version import __version__
