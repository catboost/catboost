Testpath is a collection of utilities for Python code working with files and commands.

It contains functions to check things on the filesystem, and tools for mocking
system commands and recording calls to those.

`Documentation on ReadTheDocs <https://testpath.readthedocs.io/en/latest/>`_

e.g.::

    import testpath
    testpath.assert_isfile(path)
    
    with testpath.assert_calls('git', ['add', path]):
        function_under_test()
