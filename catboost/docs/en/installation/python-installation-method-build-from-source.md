# Build Python package from source

{% include [supported-versions](../_includes/work_src/reusage-installation/python__supported-versions.md) %}

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

1. `build` Python package.

1. `setuptools` Python package, version 64.0+. Installed by default for Python < 3.12, an explicit installation is needed for Python 3.12+.

    {% cut "Previous requirements" %}

    For revisions before [5c26d15](https://github.com/catboost/catboost/commit/5c26d15fa5f218966a3dafb3f047d7f35650f235) supported 'setuptools' versions were >= 64.0 and < 81.0.

    {% endcut %}

1. Other setup dependencies that can be formulated as python packages are listed in [`pyproject.toml`](https://github.com/catboost/catboost/blob/master/catboost/python-package/pyproject.toml)'s `build-system.requires` and in [`setup.py`](https://github.com/catboost/catboost/blob/master/catboost/python-package/setup.py) in standard `setup_requires` parameter and processed using standard Python tools.

    {% note info %}

      For some reason Python 3.12 fails to automatically resolve build/setup dependencies in a way that they are buildable so it is recommended to install the following packages using pip explicitly:
        - setuptools
        - wheel (if using setuptools < 70.1.0, ['wheel' functionality has been integrated into setuptools since 70.1.0](https://github.com/pypa/setuptools/issues/1386) )
        - jupyterlab (3.x, 4.x is not supported yet, see [the relevant issue](https://github.com/catboost/catboost/issues/2533))
        - conan (2.4.1+, for revisions before [21a3f85](https://github.com/catboost/catboost/commit/21a3f856c118b8c2514f0307ca7b013d6329015e) only conan 1.x with versions 1.62.0+ is supported)

    {% endnote %}

1. For building CatBoost visualization widget bundled together with the python package (enabled by default) additional setup is required:
    1. [Node.js](https://nodejs.org/) installation with `npm` command accessible from the shell.
    1. [`rimraf` Node.js package](https://www.npmjs.com/package/rimraf) installed with `npm`'s `--global` option (this way `rimraf` command will be accessible from the shell).
    1. [`yarn` package manager](https://yarnpkg.com/), version from 1.x series, `1.22.10` or later. Installed with `npm`'s `--global` option (this way `yarn` command will be accessible from the shell)
  An example command to install: `npm install --global yarn@1.22.10`.

    If you don't need CatBoost visualization widget support you can disable it's building and bundling with the CatBoost python package by passing `--no-widget` build/installation option.

1. Installation dependencies are listed in [`setup.py`](https://github.com/catboost/catboost/blob/master/catboost/python-package/setup.py) in standard `install_requires` parameter and processed using standard Python tools.

1. User-defined functions

    {% include [python__user-defined-functions-dependencies](../_includes/work_src/reusage-installation/python__user-defined-functions-dependencies.md) %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

## Building

Open the `$CATBOOST_SRC_ROOT/catboost/python-package` directory from the local copy of the {{ product }} repository.

Use Python's standard procedures:

### Build the wheel distribution {#build-wheel}

{% note warning %}

Note that built Python wheels will be compatible only with:
- the same platform that you build them on (i.e. `linux-x86_64`, `macos-arm64` etc.). Cross-compilation on Linux and building universal2 - compatible packages on macOS are possible but complicated, you can look at [ci/build_all.py script](https://github.com/catboost/catboost/blob/master/ci/build_all.py) for details.
- the same Python X.Y versions as the Python interpreter that you run the build command with.

{% endnote %}

```
python -m build --wheel --config-setting=--global-option=bdist_wheel <bdist_wheel options>
```

`bdist_wheel` options should be specified in the following way: `--config-setting=--global-option=--<flag_option>` or `--config-setting=--global-option=--<option_key>=<option_value>`

Example:

```
python -m build --wheel --config-setting=--global-option=bdist_wheel --config-setting=--global-option=--with-hnsw --config-setting=--global-option=--prebuilt-extensions-build-root-dir=/home/user/catboost/build/
```

You can also use older non-[PEP517](https://peps.python.org/pep-0517/) compliant way to build wheels:

```
python setup.py bdist_wheel <options>
```

But it is deprecated and this command does not work properly for CatBoost on recent macOS versions (14+).

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

{% note info %}

If `CUDA_PATH` or `CUDA_ROOT` environment variable is defined and contains a path to a valid `CUDA` installation, then CatBoost python package will be built with this `CUDA` version.
Otherwise `CUDA` support will be disabled in the package.

{% endnote %}
