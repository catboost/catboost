from __future__ import absolute_import
from __future__ import print_function

import ast

from _pytest.assertion import rewrite

from __res import importer


class AssertionRewritingHook(rewrite.AssertionRewritingHook):

    def find_module(self, name, path=None):
        state = self.config._assertstate
        if not self._should_rewrite(name, state):
            return None
        state.trace("find_module called for: %s" % name)

        try:
            if self.is_package(name):
                return None
        except ImportError:
            return None

        self._rewritten_names.add(name)

        state.trace("rewriting %s" % name)
        co = _rewrite_test(self.config, name)
        if co is None:
            # Probably a SyntaxError in the test.
            return None
        self.modules[name] = co, None
        return self

    def _should_rewrite(self, name, state):
        if name.startswith("__tests__.") or name.endswith(".conftest"):
            return True

        return self._is_marked_for_rewrite(name, state)

    def is_package(self, name):
        return importer.is_package(name)

    def get_source(self, name):
        return importer.get_source(name)


def _rewrite_test(config, name):
    """Try to read and rewrite *fn* and return the code object."""
    state = config._assertstate

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
    rewrite.rewrite_asserts(tree, path, config)
    try:
        co = compile(tree, path, "exec", dont_inherit=True)
    except SyntaxError:
        # It's possible that this error is from some bug in the
        # assertion rewriting, but I don't know of a fast way to tell.
        state.trace("failed to compile: %r" % (path,))
        return None
    return co
