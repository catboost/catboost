# Install the released version

{% include [reusage-cli-releases-page](../_includes/work_src/reusage-cli/releases-page.md) %}


{% note info %}

{% include [reusage-installation-gpu-support-from-the-box__p](../_includes/work_src/reusage-installation/gpu-support-from-the-box__p.md) %}

{% include [installation-compute-capability-requirements](../_includes/work_src/reusage-code-examples/compute-capability-requirements.md) %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}

To install the released {{ r-package }} binary:
1. Run the following commands:
    ```
    install.packages('remotes')
    remotes::install_url('BINARY_URL'[, INSTALL_opts = c("--no-multiarch", "--no-test-load")])
    ```

    - `BINARY_URL` is the URL of the released binary.

       It should look like this:

        `https://github.com/catboost/catboost/releases/download/v{RELEASE_VERSION}/catboost-R-{OS}-{ARCHITECTURE}-{RELEASE_VERSION}.tgz`

        |Operating system|{OS} value|Possible {ARCHITECTURE} values|GPU support using [CUDA](https://developer.nvidia.com/cuda-zone)|
        |--------|-----------------|------------|------------|
        | Linux (compatible with [manylinux2014 platform tag](https://peps.python.org/pep-0599/) ) | `linux` | `x86_64` and `aarch64` |yes|
        | macOS (versions currently supported by Apple) | `darwin` |`universal2` - supports both `x86_64` and `arm64` (Apple silicon)|no|
        | Windows 10 and 11 | `windows` | `x86_64` |yes|

    - `INSTALL_opts`:

       - `--no-multiarch` is an optional parameter for multiarch support.
       - `--no-test-load` is an argument which suppresses the testing phase.

    Examples:
    - install the macOS version {{ release-version }} of the R package:
    ```
    install.packages('remotes')
    remotes::install_url('https://github.com/catboost/catboost/releases/download/v{{ release-version }}/catboost-R-darwin-universal2-{{ release-version }}.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
    ```

    - install the Linux version {{ release-version }} of the R package for the `aarch64` CPU architecture:
    ```
    install.packages('remotes')
    remotes::install_url('https://github.com/catboost/catboost/releases/download/v{{ release-version }}/catboost-R-linux-aarch64-{{ release-version }}.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
    ```

    - install the Windows version {{ release-version }} of the R package for the `x86_64` CPU architecture:
    ```
    install.packages('remotes')
    remotes::install_url('https://github.com/catboost/catboost/releases/download/v{{ release-version }}/catboost-R-windows-x86_64-{{ release-version }}.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
    ```

    {% note warning %}

    Do not forget to change the package version to the required one. The example above illustrates the installation of the version {{ release-version }}.

    To install another version, let's say 0.16.5, change the code to the following:
    ```
    install.packages('remotes')
    remotes::install_url('https://github.com/catboost/catboost/releases/download/v0.16.5/catboost-R-Windows-0.16.5.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
    ```

    {% endnote %}


{% include [r__troubleshooting](../_includes/work_src/reusage-installation/r__troubleshooting.md) %}
