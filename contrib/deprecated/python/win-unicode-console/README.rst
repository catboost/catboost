
win-unicode-console
===================

A Python package to enable Unicode input and display when running Python from Windows console.

General information
-------------------

When running Python in the standard console on Windows, there are several problems when one tries to enter or display Unicode characters. The relevant issue is http://bugs.python.org/issue1602. This package solves some of them.

- First, when you want to display Unicode characters in Windows console, you have to select a font able to display them. Similarly, if you want to enter Unicode characters, you have to have you keyboard properly configured. This has nothing to do with Python, but is included here for completeness.
  
- The standard stream objects (``sys.stdin``, ``sys.stdout``, ``sys.stderr``) are not capable of reading and displaying Unicode characters in Windows console. This has nothing to do with encoding, since even ``sys.stdin.buffer.raw.readline()`` returns ``b"?\n"`` when entering ``α`` and there is no encoding under which ``sys.stdout.buffer.raw.write`` displays ``α``.
  
  The ``streams`` module provides several alternative stream objects. ``stdin_raw``, ``stdout_raw``, and ``stderr_raw`` are raw stream objects using WinAPI functions ``ReadConsoleW`` and ``WriteConsoleW`` to interact with Windows console through UTF-16-LE encoded bytes. The ``stdin_text``, ``stdout_text``, and ``stderr_text`` are standard text IO wrappers over standard buffered IO over our raw streams, and are intended to be primary replacements to ``sys.std*`` streams. Unfortunately, other wrappers around ``std*_text`` are needed (see below), so there are more stream objects in ``streams`` module.
  
  The function ``streams.enable`` installs chosen stream objects instead of the original ones. By default, it chooses appropriate stream objects itself. The function ``streams.disable`` restores the original stream objects (these are stored in ``sys.__std*__`` attributes by Python).
  
  After replacing the stream objects, also using ``print`` with a string containing Unicode characters and displaying Unicode characters in the interactive loop works. For ``input``, see below.
  
- Python interactive loop doesn't use ``sys.stdin`` to read input so fixing it doesn't help. Also the ``input`` function may or may not use ``sys.stdin`` depending on whether ``sys.stdin`` and ``sys.stdout`` have the standard filenos and whether they are interactive. See http://bugs.python.org/issue17620 for more information.
  
  To solve this, we install a custom readline hook. Readline hook is a function which is used to read a single line interactively by Python REPL. It may also be used by ``input`` function under certain conditions (see above). On Linux, this hook is usually set to GNU readline function, which provides features like autocompletion, history, …
  
  The module ``readline_hook`` provides our custom readline hook, which uses ``sys.stdin`` to get the input and is (de)activated by functions ``readline_hook.enable``, ``readline_hook.disable``. 
  
  As we said, readline hook can be called from two places – from the REPL and from ``input`` function. In the first case the prompt is encoded using ``sys.stdin.encoding``, but in the second case ``sys.stdout.encoding`` is used. So Python currently makes an assumption that these two encodings are equal.
  
- Python tokenizer, which is used when parsing the input from REPL, cannot handle UTF-16 or generally any encoding containing null bytes. Because UTF-16-LE is the encoding of Unicode used by Windows, we have to additionally wrap our text stream objects (``std*_text``). Thus, ``streams`` module contains also stream objects ``stdin_text_transcoded``, ``stdout_text_transcoded``, and ``stderr_text_transcoded``. They basically just hide the underlying UTF-16-LE encoded buffered IO, and sets encoding to UTF-8. These transcoding wrappers are used by default by ``streams.enable``.

There are additional issues on Python 2.

- Since default Python 2 strings correspond to ``bytes`` rather than ``unicode``, people are usually calling ``print`` with ``bytes`` argument. Therefore, ``sys.stdout.write`` and ``sys.stderr.write`` should support ``bytes`` argument. That is why we add ``stdout_text_str`` and ``stderr_text_str`` stream objects to ``streams`` module. They are used by default on Python 2.
  
- When we enter a Unicode literal into interactive interpreter, it gets processed by the Python tokenizer, which is bytes-based. When we enter ``u"\u03b1"`` into the interactive interpreter, the tokenizer gets essentially ``b'u"\xce\xb1"'`` plus the information that the encoding used is UTF-8. The problem is that the tokenizer uses the encoding only if ``sys.stdin`` is a file object (see https://hg.python.org/cpython/file/d356e68de236/Parser/tokenizer.c#l797). Hence, we introduce another stream object ``streams.stdin_text_fileobj`` that wraps ``stdin_text_transcoded`` and also is structurally compatible with Python file object. This object is used by default on Python 2.
  
- The check for interactive streams done by ``raw_input`` unfortunately requires that both ``sys.stdin`` and ``sys.stdout`` are file objects. Besides ``stdin_text_fileobj`` for stdin we could use also ``stdout_text_str_fileobj`` for stdout. Unfortunately, that breaks ``print``.
  
  Using ``print`` statement or function leads to calling ``PyFile_WriteObject`` with ``sys.stdout`` as argument. Unfortunately, its generic ``write`` method is used only if it is *not* a file object. Otherwise, ``PyObject_Print`` is called, and this function is file-based, so it ends with a ``fprintf`` call, which is not something we want. In conclusion, we need stdout *not* to be a file object.
  
  Given the situation described, the best solution seems to be reimplementing ``raw_input`` and ``input`` builtin functions and monkeypatching ``__builtins__``. This is done by our ``raw_input`` module on Python 2.

- Similarly to the input from from ``sys.stdin`` the arguments in ``sys.argv`` are also ``bytes`` on Python 2 and the original ones may not be reconstructable. To overcome this we add ``unicode_argv`` module. The function ``unicode_argv.get_unicode_argv`` returns Unicode version of ``sys.argv`` obtained by WinAPI functions ``GetCommandLineW`` and ``CommandLineToArgvW``. The function ``unicode_argv.enable`` monkeypatches ``sys.argv`` with the Unicode arguments.


Installation
------------

Install the package from PyPI via ``pip install win-unicode-console`` (recommended), or download the archive and install it from the archive (e.g. ``pip install win_unicode_console-0.x.zip``), or install the package manually by placing directory ``win_unicode_console`` and module ``run.py`` from the archive to the ``site-packages`` directory of your Python installation.


Usage
-----

The top-level ``win_unicode_console`` module contains a function ``enable``, which install various fixes offered by ``win_unicode_console`` modules, and a function ``disable``, which restores the original environment. By default, custom stream objects are installed as well as a custom readline hook. On Python 2, ``raw_input`` and ``input`` functions are monkeypatched. ``sys.argv`` is not monkeypatched by default since unfortunately some Python 2 code strictly assumes ``str`` instances in ``sys.argv`` list. Use ``enable(use_unicode_argv=True)`` if you want the monkeypathcing. For further customization, see the sources. The logic should be clear.

Generic usage of the package is just calling ``win_unicode_console.enable()`` whenever the fixes should be applied and ``win_unicode_console.disable()`` to revert all the changes. Note that it should be a responsibility of a Python user on Windows to install ``win_unicode_console`` and fix his Python environment regarding Unicode interaction with console, rather than of a third-party developer enabling ``win_unicode_console`` in his application, which adds a dependency. Our package should be seen as an external patch to Python on Windows rather than a feature package for other packages not directly related to fixing Unicode issues.

Different ways of how ``win_unicode_console`` can be used to fix a Python environment on Windows follow.

- *Python patch (recommended).* Just call ``win_unicode_console.enable()`` in your ``sitecustomize`` or ``usercustomize`` module (see https://docs.python.org/3/tutorial/appendix.html#the-customization-modules for more information). This will enable ``win_unicode_console`` on every run of the Python interpreter (unless ``site`` is disabled). Doing so should not break executed scripts in any way. Otherwise, it is a bug of ``win_unicode_console`` that should be fixed.

- *Opt-in runner.* You may easily run a script with ``win_unicode_console`` enabled by using our ``runner`` module and its helper ``run`` script. To do so, execute ``py -i -m run script.py`` instead of ``py -i script.py`` for interactive mode, and similarly ``py -m run script.py`` instead of ``py script.py`` for non-interactive mode. Of course you may provide arguments to your script: ``py -i -m run script.py arg1 arg2``. To run the bare interactive interpreter with ``win_unicode_console`` enabled, execute ``py -i -m run``.

- *Opt-out runner.* In case you are using ``win_unicode_console`` as Python patch, but you want to run a particular script with ``win_unicode_console`` disabled, you can also use the runner. To do so, execute ``py -i -m run --init-disable script.py``.

- *Customized runner.* To move arbitrary initialization (e.g. enabling ``win_unicode_console`` with non-default arguments) from ``sitecustomize`` to opt-in runner, move it to a separate module and use ``py -i -m run --init-module module script.py``. That will import a module ``module`` on startup instead of enabling ``win_unicode_console`` with default arguments.


Compatibility
-------------

``win_unicode_console`` package was tested on Python 3.4, Python 3.5, and Python 2.7. 32-bit or 64-bit shouldn't matter. It also interacts well with the following packages:

- ``colorama`` package (https://pypi.python.org/pypi/colorama) makes ANSI escape character sequences (for producing colored terminal text and cursor positioning) work under MS Windows. It does so by wrapping ``sys.stdout`` and ``sys.stderr`` streams. Since ``win_unicode_console`` replaces the streams in order to support Unicode, ``win_unicode_console.enable`` has to be called before ``colorama.init`` so everything works as expected.
  
  As of ``colorama`` v0.3.3, there was an early binding issue (https://github.com/tartley/colorama/issues/32), so ``win_unicode_console.enable`` has to be called even before importing ``colorama``. Note that is already the case when ``win_unicode_console`` is used as Python patch or as opt-in runner. The issue was already fixed.

- ``pyreadline`` package (https://pypi.python.org/pypi/pyreadline/2.0)  implements GNU readline features on Windows. It provides its own readline hook, which actually supports Unicode input. ``win_unicode_console.readline_hook`` detects when ``pyreadline`` is active, and in that case, by default, reuses its readline hook rather than installing its own, so GNU readline features are preserved on top of our Unicode streams.

- ``IPython`` (https://pypi.python.org/pypi/ipython) can be also used  with ``win_unicode_console``.
  
  As of ``IPython`` 3.2.1, there is an early binding issue (https://github.com/ipython/ipython/issues/8669), so ``win_unicode_console.enable`` has to be called even before importing ``IPython``. That is the case when ``win_unicode_console`` is used as Python patch.
  
  There was also an issue that IPython was not compatible with the builtin function ``raw_input`` returning unicode on Python 2 (https://github.com/ipython/ipython/issues/8670). If you hit this issue, you can make ``win_unicode_console.raw_input.raw_input`` return bytes by enabling it as ``win_unicode_console.enable(raw_input__return_unicode=False)``. This was fixed in IPython 4.


Backward incompatibility
------------------------

- Since version 0.4, the signature of ``streams.enable`` has been changed because there are now more options for the stream objects to be used. It now accepts a keyword argument for each ``stdin``, ``stdout``, ``stderr``, setting the corresponding stream. ``None`` means “do not set”, ``Ellipsis`` means “use the default value”.
  
  A function ``streams.enable_only`` was added. It works the same way as ``streams.enable``, but the default value for each parameter is ``None``.
  
  Functions ``streams.enable_reader``, ``streams.enable_writer``, and ``streams.enable_error_writer`` have been removed. Example: instead of ``streams.enable_reader(transcode=True)`` use ``streams.enable_only(stdin=streams.stdin_text_transcoding)``.
  
  There are also corresponding changes in top-level ``enable`` function.
  
- Since version 0.3, the custom stream objects have the standard filenos, so calling ``input`` doesn't handle Unicode without custom readline hook.


Acknowledgements
----------------

- The code of ``streams`` module is based on the code submitted to http://bugs.python.org/issue1602.
- The idea of providing custom readline hook and the code of ``readline_hook`` module is based on https://github.com/pyreadline/pyreadline.
- The code related to ``unicode_argv.get_full_unicode_argv`` is based on http://code.activestate.com/recipes/572200/.
- The idea of using path hooks and the code related to ``unicode_argv.argv_setter_hook`` is based on https://mail.python.org/pipermail/python-list/2016-June/710183.html.
