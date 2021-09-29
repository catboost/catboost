# Install from a local copy on Windows

{% note info %}

- {% include [r-package-r__recommended-installation-method](../_includes/work_src/reusage-installation/r__recommended-installation-method.md) %}

- {% include [installation-windows-visual-cplusplus-required](../_includes/work_src/reusage-code-examples/windows-visual-cplusplus-required.md) %}

  {% if audience == "internal" %} {% include [arc_users](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}

- {% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


To install the {{ r-package }} from a local copy of the {{ product }} repository on Windows:

1. [Download](https://git-scm.com/download/) and install Git locally.

1. Open Git Bash and create a directory for the local copy of the {{ product }} repository:
    ```no-highlight
    mkdir CatBoostRepository
    ```

1. Open the created directory.

1. (Optionally) Configure the proxy if required:
    ```no-highlight
    git config --global http.proxy http://login:password@ip_address:port_number
    ```

    - `login` and `password` are the proxy credentials
    - `ip_address` is the IP address of the proxy server
    - `port_number` is the configured proxy port number

1. Clone the repository:

    ```no-highlight
    {{ installation--git-clone }}
    ```

1. Install Python and add it to PATH.

1. {% if audience == "internal" %} {% include [arcadia_users_step](../yandex_specific/_includes/arcadia_users_step.md) %} {% endif %}Install [Visual Studio Community 2017](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017).

    - Choose the **Windows Platform development** and **Desktop Development with C++** options in the **workloads** tab.
    - Choose a suitable version of the MSVC compiler. The version of MSVC for CUDA Toolkit 9.0 and 9.1 should not be higher than 15.6 v14.11, while the version for CUDA Toolkit 9.2 should not be higher than 15.6 v14.13.

    {% note info %}

    Visual Studio forcibly installs the latest version of the compiler upon each update. The latest MSVC compiler may not be suitable for compiling {{ product }}, especially with CUDA. It is advised to install VC++ 2017 version 15.6 v14.13 for CUDA Toolkit 9.2 and 15.6 v14.11 for CUDA Toolkit 9.0 and 9.1.

    {% cut "Identify the set version of the compiler" %}


    1. Open the properties window for any `cpp` file of the project.
    1. Ensure the absence of the `/nologo` option in the compiler's command-line (for example, by adding the deprecated `/nologo-` option in the **Command Line/Additional Options** box).
    1. Compile this source file (**Ctrl** + **F7**).

    The set version of the compiler is printed to the **Output window**. {{ product }} can not be compiled with 19.14.* versions.

    {% endcut %}

    {% cut "Change the version of the compiler" %}

    Use one of the following methods to set the recommended version of the compiler:
    - Enable the required version as described in the [Visual C++ Team Blog](https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017/).

    - Run the environment setter from the command line with the `vcvars_ver` option (the path to the script depends on the installation settings):

    ```bash
    call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.13
    ```

    Then open the solution:
    ```no-highlight
    start msvs\arcadia.sln
    ```
    {% endcut %}

    {% endnote %}

1. Perform the following steps in RStudio:
    1. Check for package updates: .
    1. Open the `catboost/catboost/R-package` directory from the local copy of the {{ product }} repository.

    For example, if the cloned repository local path is `C:\CatBoostRepository` the following command should be used:
    ```
    setwd("C:/CatBoostRepository/catboost/catboost/R-package")
    ```

    1. Run the following commands:

    ```no-highlight
    install.packages('devtools')
    devtools::build()
    devtools::install()
    ```


