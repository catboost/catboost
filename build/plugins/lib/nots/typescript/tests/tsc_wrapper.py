import pytest

from build.plugins.lib.nots.typescript import TsConfig, TsValidationError


def test_ts_config_validate_valid():
    cfg = TsConfig(path="/tsconfig.json")
    cfg.data = {
        "compilerOptions": {
            "rootDir": "./src",
            "outDir": "./build",
        },
    }

    cfg.validate()


def test_ts_config_validate_empty():
    cfg = TsConfig(path="/tsconfig.json")

    with pytest.raises(TsValidationError) as e:
        cfg.validate()

    assert e.value.errors == [
        "'rootDir' option is required",
        "'outDir' option is required",
    ]


def test_ts_config_validate_invalid_common():
    cfg = TsConfig(path="/tsconfig.json")
    cfg.data = {
        "compilerOptions": {
            "preserveSymlinks": True,
            "rootDirs": [],
            "outFile": "./foo.js",
        },
        "references": [],
        "files": [],
        "include": [],
        "exclude": [],
    }

    with pytest.raises(TsValidationError) as e:
        cfg.validate()

    assert e.value.errors == [
        "'rootDir' option is required",
        "'outDir' option is required",
        "'outFile' option is not supported",
        "'preserveSymlinks' option is not supported due to pnpm limitations",
        "'rootDirs' option is not supported, relative imports should have single root",
        "'files' option is not supported, use 'include'",
        "composite builds are not supported, use peerdirs in ya.make instead of 'references' option",
    ]


def test_ts_config_validate_invalid_subdirs():
    cfg = TsConfig(path="/foo/tsconfig.json")
    cfg.data = {
        "compilerOptions": {
            "rootDir": "/bar/src",
            "outDir": "../bar/build",
        },
    }

    with pytest.raises(TsValidationError) as e:
        cfg.validate()

    assert e.value.errors == [
        "'rootDir' should be a subdirectory of the module",
        "'outDir' should be a subdirectory of the module",
    ]


def test_ts_config_transform():
    cfg = TsConfig(path="/tsconfig.json")
    cfg.data = {
        "compilerOptions": {
            "rootDir": "./src",
            "outDir": "./build",
            "typeRoots": ["./node_modules/foo", "bar"],
        },
        "include": ["src/**/*"],
    }

    cfg.transform_paths(
        build_path="bindir",
        sources_path="srcdir",
    )

    assert cfg.data == {
        "compilerOptions": {
            "outDir": "bindir/build",
            "rootDir": "srcdir/src",
            "baseUrl": "bindir/node_modules",
            "typeRoots": ["srcdir/node_modules/foo", "srcdir/bar", "bindir/node_modules/foo", "bindir/bar"]
        },
        "include": ["srcdir/src/**/*"],
        "exclude": [],
    }


def test_ts_config_transform_when_root_eq_out():
    cfg = TsConfig(path="/tsconfig.json")
    cfg.data = {
        "compilerOptions": {
            "rootDir": ".",
            "outDir": ".",
        },
    }

    cfg.transform_paths(
        build_path="bindir",
        sources_path="srcdir",
    )

    assert cfg.data == {
        "compilerOptions": {
            "rootDir": "srcdir",
            "outDir": "bindir",
            "baseUrl": "bindir/node_modules",
        },
        "include": [],
        "exclude": [],
    }


def test_ts_config_transform_sets_correct_source_root():
    cfg = TsConfig(path="/tsconfig.json")
    cfg.data = {
        "compilerOptions": {
            "rootDir": "src",
            "outDir": "build",
            "sourceMap": True,
        },
    }

    cfg.transform_paths(
        build_path="bindir",
        sources_path="srcdir",
    )

    assert cfg.data == {
        "compilerOptions": {
            "rootDir": "srcdir/src",
            "outDir": "bindir/build",
            "baseUrl": "bindir/node_modules",
            "sourceMap": True,
            "sourceRoot": "../src",
        },
        "include": [],
        "exclude": [],
    }


def test_ts_config_compiler_options():
    cfg = TsConfig(path="/tsconfig.json")

    assert cfg.compiler_option("invalid") is None

    cfg.data = {
        "compilerOptions": {
            "rootDir": "src",
        },
    }

    assert cfg.compiler_option("rootDir") == "src"
