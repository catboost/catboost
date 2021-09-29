PY23_LIBRARY(jedi)

LICENSE(MIT)



VERSION(0.13.3)

PEERDIR(
    contrib/python/parso
    contrib/python/setuptools
)

PY_SRCS(
    TOP_LEVEL
    jedi/__init__.py
    jedi/__main__.py
    jedi/_compatibility.py
    jedi/api/__init__.py
    jedi/api/classes.py
    jedi/api/completion.py
    jedi/api/environment.py
    jedi/api/exceptions.py
    jedi/api/helpers.py
    jedi/api/interpreter.py
    jedi/api/keywords.py
    jedi/api/project.py
    jedi/api/replstartup.py
    jedi/cache.py
    jedi/common/__init__.py
    jedi/common/context.py
    jedi/common/utils.py
    jedi/debug.py
    jedi/evaluate/__init__.py
    jedi/evaluate/analysis.py
    jedi/evaluate/arguments.py
    jedi/evaluate/base_context.py
    jedi/evaluate/cache.py
    jedi/evaluate/compiled/__init__.py
    jedi/evaluate/compiled/access.py
    jedi/evaluate/compiled/context.py
    jedi/evaluate/compiled/fake.py
    jedi/evaluate/compiled/getattr_static.py
    jedi/evaluate/compiled/mixed.py
    jedi/evaluate/compiled/subprocess/__init__.py
    jedi/evaluate/compiled/subprocess/__main__.py
    jedi/evaluate/compiled/subprocess/functions.py
    jedi/evaluate/context/__init__.py
    jedi/evaluate/context/asynchronous.py
    jedi/evaluate/context/function.py
    jedi/evaluate/context/instance.py
    jedi/evaluate/context/iterable.py
    jedi/evaluate/context/klass.py
    jedi/evaluate/context/module.py
    jedi/evaluate/context/namespace.py
    jedi/evaluate/docstrings.py
    jedi/evaluate/dynamic.py
    jedi/evaluate/filters.py
    jedi/evaluate/finder.py
    jedi/evaluate/flow_analysis.py
    jedi/evaluate/helpers.py
    jedi/evaluate/imports.py
    jedi/evaluate/jedi_typing.py
    jedi/evaluate/lazy_context.py
    jedi/evaluate/param.py
    jedi/evaluate/parser_cache.py
    jedi/evaluate/pep0484.py
    jedi/evaluate/recursion.py
    jedi/evaluate/stdlib.py
    jedi/evaluate/syntax_tree.py
    jedi/evaluate/sys_path.py
    jedi/evaluate/usages.py
    jedi/evaluate/utils.py
    jedi/parser_utils.py
    jedi/refactoring.py
    jedi/settings.py
    jedi/utils.py
)

RESOURCE_FILES(
    PREFIX contrib/python/jedi/
    .dist-info/METADATA
    .dist-info/top_level.txt
    jedi/evaluate/compiled/fake/_functools.pym
    jedi/evaluate/compiled/fake/_sqlite3.pym
    jedi/evaluate/compiled/fake/_sre.pym
    jedi/evaluate/compiled/fake/_weakref.pym
    jedi/evaluate/compiled/fake/builtins.pym
    jedi/evaluate/compiled/fake/datetime.pym
    jedi/evaluate/compiled/fake/io.pym
    jedi/evaluate/compiled/fake/operator.pym
    jedi/evaluate/compiled/fake/posix.pym
)

NO_LINT()

END()
