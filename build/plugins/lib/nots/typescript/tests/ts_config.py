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


def test_ts_config_compiler_options():
    cfg = TsConfig(path="/tsconfig.json")

    assert cfg.compiler_option("invalid") is None

    cfg.data = {
        "compilerOptions": {
            "rootDir": "src",
        },
    }

    assert cfg.compiler_option("rootDir") == "src"
