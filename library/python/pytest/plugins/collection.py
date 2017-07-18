import sys
import traceback

import py
import _pytest.python
import _pytest.doctest


class LoadedModule(_pytest.python.Module):

    def __init__(self, name, session):
        self.name = name + ".py"  # pytest requires names to be ended with .py
        self.session = session
        self.config = session.config
        self.fspath = py.path.local()
        self.parent = session
        self.keywords = {}

    @property
    def nodeid(self):
        return self.name

    def _getobj(self):
        module_name = "__tests__.{}".format(self.name[:-(len(".py"))])
        __import__(module_name)
        return sys.modules[module_name]


class DoctestItem(_pytest.doctest.DoctestItem):
    def repr_failure(self, excinfo):
        import doctest
        if excinfo.errisinstance((doctest.DocTestFailure, doctest.UnexpectedException)):
            doctestfailure = excinfo.value
            example = doctestfailure.example
            test = doctestfailure.test
            filename = test.filename
            if test.lineno is None:
                lineno = None
            else:
                lineno = test.lineno + example.lineno + 1
            message = excinfo.type.__name__
            reprlocation = _pytest.doctest.ReprFileLocation(filename, lineno, message)
            checker = doctest.OutputChecker()
            lines = []
            indent = '>>>'
            for line in example.source.splitlines():
                lines.append('%s %s' % (indent, line))
                indent = '...'
            if excinfo.errisinstance(doctest.DocTestFailure):
                lines += checker.output_difference(example, doctestfailure.got, doctest.REPORT_UDIFF).split("\n")
            else:
                inner_excinfo = py.code.ExceptionInfo(excinfo.value.exc_info)
                lines += ["UNEXPECTED EXCEPTION: %s" % repr(inner_excinfo.value)]
                lines += traceback.format_exception(*excinfo.value.exc_info)
            return _pytest.doctest.ReprFailDoctest(reprlocation, lines)
        else:
            return super(DoctestItem, self).repr_failure(excinfo)


class DoctestModule(LoadedModule):

    def collect(self):
        import doctest
        finder = doctest.DocTestFinder()
        module = self._getobj()
        optionflags = _pytest.doctest.get_optionflags(self)
        runner = doctest.DebugRunner(verbose=0, optionflags=optionflags)
        for test in finder.find(module, self.name[:-(len(".py"))]):
            if test.examples:  # skip empty doctests
                yield DoctestItem(test.name, self, runner, test)


class CollectionPlugin(object):
    def __init__(self, test_modules):
        self._test_modules = test_modules

    def pytest_sessionstart(self, session):

        def collect(*args, **kwargs):
            for test_module in self._test_modules:
                yield LoadedModule(test_module, session=session)
                yield DoctestModule(test_module, session=session)

        session.collect = collect
