PY_LIBRARY()



# Version: 34.3.2

PEERDIR(
    contrib/python/appdirs
    contrib/python/packaging
    contrib/python/six
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    easy_install.py
    pkg_resources/__init__.py
    setuptools/__init__.py
    setuptools/archive_util.py
    setuptools/command/__init__.py
    setuptools/command/alias.py
    setuptools/command/bdist_egg.py
    setuptools/command/bdist_rpm.py
    setuptools/command/bdist_wininst.py
    setuptools/command/build_clib.py
    setuptools/command/build_ext.py
    setuptools/command/build_py.py
    setuptools/command/develop.py
    setuptools/command/easy_install.py
    setuptools/command/egg_info.py
    setuptools/command/install.py
    setuptools/command/install_egg_info.py
    setuptools/command/install_lib.py
    setuptools/command/install_scripts.py
    setuptools/command/py36compat.py
    setuptools/command/register.py
    setuptools/command/rotate.py
    setuptools/command/saveopts.py
    setuptools/command/sdist.py
    setuptools/command/setopt.py
    setuptools/command/test.py
    setuptools/command/upload.py
    setuptools/command/upload_docs.py
    setuptools/config.py
    setuptools/dep_util.py
    setuptools/depends.py
    setuptools/dist.py
    setuptools/extension.py
    setuptools/glob.py
    setuptools/launch.py
    setuptools/lib2to3_ex.py
    setuptools/monkey.py
    setuptools/msvc.py
    setuptools/namespaces.py
    setuptools/package_index.py
    setuptools/py26compat.py
    setuptools/py27compat.py
    setuptools/py31compat.py
    setuptools/py33compat.py
    setuptools/py36compat.py
    setuptools/sandbox.py
    setuptools/site-patch.py
    setuptools/ssl_support.py
    setuptools/unicode_utils.py
    setuptools/version.py
    setuptools/windows_support.py
)

END()
