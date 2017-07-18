import re
import sys
import traceback

from __res import importer


def check_imports(no_check=None, extra=[], skip_func=None):
    """
    tests all bundled modules are importable
    just add
    "PEERDIR(library/python/import_test)" to your CMakeLists.txt and
    "from import_test import test_imports" to your python test source file.
    """
    exceptions = [
        '__yt_entry_point__',

        'boto.*',

        'click._winconsole',
        'common.*',  # sandbox.common

        'flask.ext.__init__',

        'gensim.models.lda_worker',
        'gensim.models.lda_dispatcher',
        'gensim.models.lsi_dispatcher',
        'gensim.models.lsi_worker',
        'gensim.similarities.index',

        'kernel.*',  # skynet/kernel
        'IPython.*',
        'lxml.cssselect',
        'lxml.html.ElementSoup',
        'lxml.html.diff',
        'lxml.html._diffcommand',
        'lxml.html._html5builder',
        'lxml.html.html5parser',
        'lxml.html.soupparser',
        'lxml.html.usedoctest',
        'lxml.isoschematron.__init__',
        'lxml.usedoctest',
        'py._code._assertionnew',
        'py._code._assertionold',
        'rbtools.clients.tests',
        'requests.__init__',
        'requests.packages.chardet.chardetect',
        'requests.packages.urllib3.contrib.ntlmpool',
        'setuptools.*',
        '_pytest.assertion.newinterpret',
        '__tests__.*',  # all test modules get imported when tests are run

        "yt.packages.*",
        "tornado.platform.*",
        "tornado.curl_httpclient",
        "google.protobuf.internal.cpp_message",
        "google.protobuf.pyext.cpp_message",
        "numpy.distutils.*",
        "numpy.core.setup",
        "numpy.core.cversions",
        "numpy.lib.__init__",
        "numpy.doc.*",
        "numpy.testing.__init__",
        "numpy.ma.version",
        "numpy.matrixlib.__init__",
        "numpy.testing.noseclasses",
        "numpy.__init__",
        "numpy.core.code_generators.generate_numpy_api",
        "numpy.doc.basics",
        "numpy.doc.broadcasting",
        "scipy.misc.__init__",  # XXX: fixme
        "kazoo.handlers.*",
        "psutil._pssunos",
        "psutil._psbsd",
        "psutil._psosx",
        "psutil._pswindows",
        "psutil._psutil_common",
        "psutil._psutil_bsd",
        "psutil._psutil_osx",
        "psutil._psutil_sunos",
        "psutil._psutil_windows",

        "PIL.ImageCms",
        "PIL.ImageGL",
        "PIL.ImageGrab",
        "PIL.ImageQt",
        "PIL.ImageTk",

        "coloredlogs.cli",

        "dateutil.tzwin",

        "matplotlib.backends.*",
        "matplotlib.sphinxext.*",
        "matplotlib.testing.*",
        "mpl_toolkits.*",

        "networkx.algorithms.connectivity.__init__",
        "networkx.algorithms.flow.__init__",
        "networkx.testing.__init__",

        "nile.*",

        "pandas.io.auth",
        "pandas.io.data",
        "pandas.io.ga",
        "pandas.io.s3",
        "pandas.io.wb",
        "pandas.rpy.base",
        "pandas.rpy.vars",
        "pandas.util.clipboard",

        "parsel.unified",

        "ptpython.contrib.asyncssh_repl",
        "ptpython.ipython",

        "prompt_toolkit.clipboard.pyperclip",
        "prompt_toolkit.eventloop.asyncio_posix",
        "prompt_toolkit.eventloop.asyncio_win32",
        "prompt_toolkit.eventloop.win32",
        "prompt_toolkit.terminal.conemu_output",
        "prompt_toolkit.win32_types",
        "prompt_toolkit.terminal.win32_input",
        "prompt_toolkit.terminal.win32_output",

        "backports.__init__",
        "pygments.sphinxext",

        "qb2.doc.extractor_graph",
        "qb2.doc.regex",
        "qb2.doc.resources",

        "raven.contrib.*",
        "raven.handlers.logbook",
        "raven.utils.testutils",

        "subvertpy.ra_svn",  # can only be imported after subvertpy.ra
        "superfcgi.*",

        "thrift.TSCons",
        "thrift.TTornado",
        "thrift.transport.*",
        "twisted.*",

        "uwsgidecorators",

        "watchdog.*",
        "werkzeug.*",
        "ws4py.*",
        "services.lfm.*",

        "sqlalchemy.testing",
        "gevent.win32util",

        "library.python.ctypes.__init__",

    ] + list(no_check or [])

    patterns = [re.escape(s).replace(r'\*', r'.*') for s in exceptions]
    rx = re.compile('^({})$'.format('|'.join(patterns)))

    failed = []

    for module in sys.extra_modules:
        if rx.search(module):
            continue

        if skip_func and skip_func(module):
            continue

        if module == '__main__' and 'if __name__ ==' not in importer.get_data(module):
            print 'SKIP:', module, '''without "if __name__ == '__main__'" check'''
            continue

        try:
            print 'TRY:', module
            if module == '__main__':
                importer.load_module('__main__', '__main__py')
            else:
                __import__(module)
            print 'OK:', module

        except Exception as e:
            print >>sys.stderr, 'FAIL:', module, e
            traceback.print_exception(*sys.exc_info())
            failed.append('{}: {}'.format(module, e))

    if failed:
        raise ImportError('modules not imported:\n' + '\n'.join(failed))


test_imports = check_imports


def main():
    skip_names = sys.argv[1:]
    print "Skip patterns:", skip_names

    try:
        check_imports(no_check=skip_names)
    except:
        sys.exit(1)
