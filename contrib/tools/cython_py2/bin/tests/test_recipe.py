import ast
import py_compile
import re
from pathlib import Path

import pytest
import yatest.common as yc


def test_dependency_inventory_ignores_compiler_bytecode(tmp_path):
    recipe = Path(yc.source_path("contrib/tools/cython_py2/.yandex_meta/__init__.py"))
    # Load the inventory helpers without starting the Nix importer.
    tree = ast.parse(recipe.read_text())
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"write_dependencies", "runtime_files"}
    ]
    namespace = {"Path": Path, "re": re}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(recipe), "exec"), namespace)

    includes = tmp_path / "Cython/Includes"
    utility = tmp_path / "Cython/Utility"
    includes.mkdir(parents=True)
    utility.mkdir()
    (includes / "example.pxd").write_text('cdef extern from "stdint.h":\n    pass\n')
    source = utility / "__init__.py"
    source.write_text("# Imported while building Cython.\n")

    generate = namespace["write_dependencies"]
    generate(tmp_path, "contrib/tools/cython_py2")
    expected = [(tmp_path / name).read_bytes() for name in ["induced_deps.h", "sysincl.yml"]]
    expected_runtime = namespace["runtime_files"](tmp_path)

    py_compile.compile(str(source), doraise=True)
    assert list(utility.rglob("*.pyc"))
    generate(tmp_path, "contrib/tools/cython_py2")
    assert [(tmp_path / name).read_bytes() for name in ["induced_deps.h", "sysincl.yml"]] == expected
    assert namespace["runtime_files"](tmp_path) == expected_runtime


def test_build_sysincl_matches_supplier_inventory():
    mirror_reference = "build/sysincl/cython-py2.yml"
    mirror = Path(yc.source_path(mirror_reference))
    if not mirror.is_file():
        configuration = Path(yc.source_path("build/conf/sysincl.conf")).read_text()
        assert mirror_reference not in configuration, "The build configuration references a missing SYSINCL file"
        pytest.skip("The build configuration has not adopted this supplier manifest yet")

    inventory = Path(yc.source_path("contrib/tools/cython_py2/sysincl.yml"))
    assert mirror.read_bytes() == inventory.read_bytes(), "The build SYSINCL copy differs from the supplier inventory"
