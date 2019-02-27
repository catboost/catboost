from __future__ import print_function

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

        'flaky.flaky_pytest_plugin',

        'flask.ext.__init__',
        'future.backports.email.policy',  # email backport is incomplete in v0.16.0.
        'future.moves.dbm.ndbm',

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
        '_pytest.*',
        '__tests__.*',  # all test modules get imported when tests are run

        "yt.packages.*",
        "yt.wrapper.cypress_fuse",
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
        "dateutil.tz.win",

        "matplotlib.backends.*",
        "matplotlib.sphinxext.*",
        "matplotlib.testing.*",
        "mpl_toolkits.*",

        "mwlib.*",

        "networkx.algorithms.connectivity.__init__",
        "networkx.algorithms.flow.__init__",
        "networkx.testing.__init__",

        "nile.drivers.yql.*",
        "nile.style.jupyter_monitor",

        "pandas.io.auth",
        "pandas.io.data",
        "pandas.io.ga",
        "pandas.io.s3",
        "pandas.io.wb",
        "pandas.rpy.base",
        "pandas.rpy.common",
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

        "raven.contrib.*",
        "raven.handlers.logbook",
        "raven.utils.testutils",

        "sklearn.utils.*",

        "subvertpy.ra_svn",  # can only be imported after subvertpy.ra
        "superfcgi.*",

        "tenacity.tornadoweb",
        "thrift.TSCons",
        "thrift.TTornado",
        "thrift.transport.*",
        "twisted.*",

        "uwsgidecorators",

        "watchdog.*",
        "werkzeug.*",
        "ws4py.*",

        'wtforms.ext.django.*',

        "services.lfm.*",

        "sqlalchemy.testing",
        "gevent.win32util",

        "library.python.ctypes.__init__",
        "celery.events.cursesmon",
        "billiard.popen_forkserver",
        "billiard.forkserver",
        "celery.contrib.sphinx",

        "flask_wtf.i18n",
        "playhouse.apsw_ext",
        "botocore.vendored.requests.packages.urllib3.contrib.pyopenssl",
        "cssutils._fetchgae",

        "catboost.widget.*",
        "kubiki.geobase",

    ] + list(no_check or [])

    if sys.version_info.major == 3:
        exceptions += [
            "antigravity",
            "lzma",
            "dbm.ndbm",
            "tkinter",
            "msvcrt",
            "msilib.*",
            "_msi",
            "winreg",
            "asyncio.test_utils",
            "dbm.gnu",
            "multiprocessing.popen_spawn_win32",
            "encodings.mbcs",
            "turtle",
            "ctypes.wintypes",
            "asyncio.windows_utils",
            "distutils.msvc9compiler",
            "distutils._msvccompiler",
            "urllib3.packages.ordered_dict",
            "encodings.oem",
            "crypt",
            "asyncio.windows_events",
            "encodings.cp65001",
            "curses.*",
            "distutils.command.bdist_msi",
            "yaml.cyaml",
            "vh.ext.nirvana.nirvana_api_bridge",
        ]

    patterns = [re.escape(s).replace(r'\*', r'.*') for s in exceptions]
    rx = re.compile('^({})$'.format('|'.join(patterns)))

    failed = []

    for module in sys.extra_modules:
        if rx.search(module):
            continue

        if skip_func and skip_func(module):
            continue

        if module == '__main__' and 'if __name__ ==' not in importer.get_source(module):
            print('SKIP:', module, '''without "if __name__ == '__main__'" check''')
            continue

        try:
            print('TRY:', module)
            if module == '__main__':
                importer.load_module('__main__', '__main__py')
            elif module.endswith('.__init__'):
                __import__(module[:-len('.__init__')])
            else:
                __import__(module)
            print('OK:', module)

        except Exception as e:
            print('FAIL:', module, e, file=sys.stderr)
            traceback.print_exception(*sys.exc_info())
            failed.append('{}: {}'.format(module, e))

    if failed:
        raise ImportError('modules not imported:\n' + '\n'.join(failed))


test_imports = check_imports


def main():
    skip_names = sys.argv[1:]
    print("Skip patterns:", skip_names)

    try:
        check_imports(no_check=skip_names)
    except:
        sys.exit(1)
