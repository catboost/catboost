# Build Python package from source

{% if audience == "internal" %}

{% include [internal-note__use-outside-arcadia](../yandex_specific/_includes/note__use-outside-arcadia.md) %}

{% endif %}

{% note warning %}

This page describes the new approach for building using standard Python tools that works since [this commit](https://github.com/catboost/catboost/commit/f37d091421089231ed3c74a0431fff1c3544d606).

For building with earlier versions see these pages:

- [In-place build on Linux and macOS](python-installation-method-build-from-source-linux-macos-using-ya-make.md)
- [In-place build on Windows](python-installation-method-build-from-source-windows-using-ya-make.md)
- [Build a wheel using mk_wheel.py](python-installation-method-build-a-wheel-package.md#mk-wheel)

{% endnote %}

## Source code

{% include [get-source-code-from-github](../_includes/work_src/reusage-installation/get-source-code-from-github.md) %}

{% include [catboost-src-root](../_includes/work_src/reusage-installation/catboost-src-root.md) %}

## Dependencies and requirements

1. As {{ product }} Python package has a native extension library as its' core [build environment setup for CMake](build-environment-setup-for-cmake.md) is required.

1. Other setup dependencies that can be formulated as python packages are listed in [`pyproject.toml`](https://github.com/catboost/catboost/blob/master/catboost/python-package/pyproject.toml)'s `build-system.requires` and in [`setup.py`](https://github.com/catboost/catboost/blob/master/catboost/python-package/setup.py) in standard `setup_requires` parameter and processed using standard Python tools.

1. For building CatBoost visualization widget bundled together with the python package (enabled by default) additional setup is required:
    1. [Node.js](https://nodejs.org/) installation with `npm` command accessible from the shell.
    1. [`rimraf` Node.js package](https://www.npmjs.com/package/rimraf) installed with `npm`'s `--global` option (this way `rimraf` command will be accessible from the shell).
    1. [`yarn` package manager](https://yarnpkg.com/), version from 1.x series, `1.22.10` or later. Installed with `npm`'s `--global` option (this way `yarn` command will be accessible from the shell)
  An example command to install: `npm install --global yarn@1.22.10`.

    If you don't need CatBoost visualization widget support you can disable it's building and bundling with the CatBoost python package by passing `--no-widget` build/installation option.

1. Installation dependencies are listed in [`setup.py`](https://github.com/catboost/catboost/blob/master/catboost/python-package/setup.py) in standard `install_requires` parameter and processed using standard Python tools.

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

## Building

Open the `$CATBOOST_SRC_ROOT/catboost/python-package` directory from the local copy of the {{ product }} repository.

Use Python's standard procedures:

### Build the wheel distribution {#build-wheel}

{% note warning %}

Note that built Python wheels will be binary compatible only with the same Python X.Y versions.

{% endnote %}

```
python setup.py bdist_wheel <options>
```

Options can be listed by calling `python setup.py bdist_wheel --help`.

One important option is `--prebuilt-extensions-build-root-dir=<path>`. It allows to use already built binary `_catboost` extension shared library. See [Build native artifacts](build-native-artifacts.md). Set this option value to `$CMAKE_BINARY_DIR`.

The resulting wheel distribution will be created in `dist/catboost-<version>-<...>.whl`

### Build the source distribution (sdist)

```
python -m build --sdist
```

The resulting source distribution will be created in `dist/catboost-<version>.tar.gz` file.

### Other useful commands

- `build_widget`. Build CatBoost widget.

  ```
  python setup.py build_widget
  ```
  Useful if widget code remains unchanged but you want to rebuild other parts. Then run `build_widget` once and then in subsequent calls to `bdist_wheel` or other commands use `--prebuilt-widget` option.

## Installation

### [Directly from the source directory](https://pip.pypa.io/en/stable/topics/local-project-installs/#regular-installs)

Builds in the process. So [build environment setup for CMake](build-environment-setup-for-cmake.md) is required.

```
python -m pip install . <options>
```

You can pass options to setup.py's `install` stage by using `--install-option` options like this:

```
python -m pip install . --install-option=--with-hnsw --install-option=--with-cuda=/usr/local/cuda-11
```

### Create [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)

Builds in the process. So [build environment setup for CMake](build-environment-setup-for-cmake.md) is required.

```
python -m pip install --editable . <options>
```

You can pass options to setup.py's `install` stage by using `--install-option` options like this:

```
python -m pip install  --install-option=--with-hnsw --install-option=--with-cuda=/usr/local/cuda-11/ --editable .
```

### Install from the built wheel

```
python -m pip install <path-to-wheel>
```

### Install from the source distribution

Builds in the process. So [build environment setup for CMake](build-environment-setup-for-cmake.md) is required.

```
python -m pip install <path-to-sdist-tar.gz>
```
