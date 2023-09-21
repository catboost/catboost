from __future__ import absolute_import
from __future__ import print_function

import ast

import py

from _pytest.assertion import rewrite

try:
    import importlib.util
except ImportError:
    pass

try:
    from pathlib import Path
except ImportError:
    pass

from __res import importer
import sys
import six


def _get_state(config):
    if hasattr(config, '_assertstate'):
        return config._assertstate
    return config._store[rewrite.assertstate_key]


class AssertionRewritingHook(rewrite.AssertionRewritingHook):
    def __init__(self, *args, **kwargs):
        self.modules = {}
        super(AssertionRewritingHook, self).__init__(*args, **kwargs)

    def find_module(self, name, path=None):
        co = self._find_module(name, path)
        if co is not None:
            return self

    def _find_module(self, name, path=None):
        state = _get_state(self.config)
        if not self._should_rewrite(name, None, state):
            return None
        state.trace("find_module called for: %s" % name)

        try:
            if self.is_package(name):
                return None
        except ImportError:
            return None

        if hasattr(self._rewritten_names, 'add'):
            self._rewritten_names.add(name)
        else:
            self._rewritten_names[name] = Path(path[0])

        state.trace("rewriting %s" % name)
        co = _rewrite_test(self.config, name)
        if co is None:
            # Probably a SyntaxError in the test.
            return None
        self.modules[name] = co, None
        return co

    def find_spec(self, name, path=None, target=None):
        co = self._find_module(name, path)
        if co is not None:
            return importlib.util.spec_from_file_location(
                name,
                co.co_filename,
                loader=self,
            )

    def _should_rewrite(self, name, fn, state):
        if name.startswith("__tests__.") or name.endswith(".conftest"):
            return True

        return self._is_marked_for_rewrite(name, state)

    def is_package(self, name):
        return importer.is_package(name)

    def get_source(self, name):
        return importer.get_source(name)

    if six.PY3:

        def load_module(self, module):
            co, _ = self.modules.pop(module.__name__)
            try:
                module.__file__ = co.co_filename
                module.__cached__ = None
                module.__loader__ = self
                module.__spec__ = importlib.util.spec_from_file_location(module.__name__, co.co_filename, loader=self)
                exec(co, module.__dict__)  # noqa
            except:  # noqa
                if module.__name__ in sys.modules:
                    del sys.modules[module.__name__]
                raise
            return sys.modules[module.__name__]

        def exec_module(self, module):
            if module.__name__ in self.modules:
                self.load_module(module)
            else:
                super(AssertionRewritingHook, self).exec_module(module)


def _rewrite_test(config, name):
    """Try to read and rewrite *fn* and return the code object."""
    state = _get_state(config)

    source = importer.get_source(name)
    if source is None:
        return None

    path = importer.get_filename(name)

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        # Let this pop up again in the real import.
        state.trace("failed to parse: %r" % (path,))
        return None
    rewrite.rewrite_asserts(tree, py.path.local(path), config)
    try:
        co = compile(tree, path, "exec", dont_inherit=True)
    except SyntaxError:
        # It's possible that this error is from some bug in the
        # assertion rewriting, but I don't know of a fast way to tell.
        state.trace("failed to compile: %r" % (path,))
        return None
    return co
