PY23_LIBRARY()

LICENSE(MIT)



VERSION(44.1.0)

PEERDIR(
    library/python/resource
)

NO_CHECK_IMPORTS(
    setuptools.*
)

NO_LINT()

PY_SRCS(
    TOP_LEVEL
    easy_install.py
    pkg_resources/__init__.py
    pkg_resources/_vendor/__init__.py
    pkg_resources/_vendor/appdirs.py
    pkg_resources/_vendor/packaging/__about__.py
    pkg_resources/_vendor/packaging/__init__.py
    pkg_resources/_vendor/packaging/_compat.py
    pkg_resources/_vendor/packaging/_structures.py
    pkg_resources/_vendor/packaging/markers.py
    pkg_resources/_vendor/packaging/requirements.py
    pkg_resources/_vendor/packaging/specifiers.py
    pkg_resources/_vendor/packaging/utils.py
    pkg_resources/_vendor/packaging/version.py
    pkg_resources/_vendor/pyparsing.py
    pkg_resources/_vendor/six.py
    pkg_resources/extern/__init__.py
    pkg_resources/py31compat.py
    setuptools/__init__.py
    setuptools/_deprecation_warning.py
    setuptools/_imp.py
    setuptools/_vendor/__init__.py
    setuptools/_vendor/ordered_set.py
    setuptools/_vendor/packaging/__about__.py
    setuptools/_vendor/packaging/__init__.py
    setuptools/_vendor/packaging/_compat.py
    setuptools/_vendor/packaging/_structures.py
    setuptools/_vendor/packaging/markers.py
    setuptools/_vendor/packaging/requirements.py
    setuptools/_vendor/packaging/specifiers.py
    setuptools/_vendor/packaging/tags.py
    setuptools/_vendor/packaging/utils.py
    setuptools/_vendor/packaging/version.py
    setuptools/_vendor/pyparsing.py
    setuptools/_vendor/six.py
    setuptools/archive_util.py
    setuptools/build_meta.py
    setuptools/command/__init__.py
    setuptools/command/alias.py
    setuptools/command/bdist_egg.py
    setuptools/command/bdist_rpm.py
    setuptools/command/bdist_wininst.py
    setuptools/command/build_clib.py
    setuptools/command/build_ext.py
    setuptools/command/build_py.py
    setuptools/command/develop.py
    setuptools/command/dist_info.py
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
    setuptools/errors.py
    setuptools/extension.py
    setuptools/extern/__init__.py
    setuptools/glob.py
    setuptools/installer.py
    setuptools/launch.py
    setuptools/lib2to3_ex.py
    setuptools/monkey.py
    setuptools/msvc.py
    setuptools/namespaces.py
    setuptools/package_index.py
    setuptools/py27compat.py
    setuptools/py31compat.py
    setuptools/py33compat.py
    setuptools/py34compat.py
    setuptools/sandbox.py
    setuptools/site-patch.py
    setuptools/ssl_support.py
    setuptools/unicode_utils.py
    setuptools/version.py
    setuptools/wheel.py
    setuptools/windows_support.py
)

RESOURCE_FILES(
    PREFIX contrib/python/setuptools/
    .dist-info/METADATA
    .dist-info/entry_points.txt
    .dist-info/top_level.txt
)

END()
