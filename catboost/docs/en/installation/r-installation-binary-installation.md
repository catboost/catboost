# Install the released version

{% include [reusage-cli-releases-page](../_includes/work_src/reusage-cli/releases-page.md) %}


{% note info %}

{% include [installation-nvidia-driver-reqs](../_includes/work_src/reusage-code-examples/nvidia-driver-reqs.md) %}

{% endnote %}


{% include [reusage-installation-gpu-support-from-the-box__p](../_includes/work_src/reusage-installation/gpu-support-from-the-box__p.md) %}


To install the released {{ r-package }} binary with GPU support:
1. Create a script file with the following code:
    ```no-highlight
    install.packages('devtools')
    devtools::install_url('BINARY_URL'[, INSTALL_opts = c("--no-multiarch", "--no-test-load")])
    ```
    
    - `BINARY_URL` is the URL of the released binary.
    
    - `INSTALL_opts`:
    
       - `--no-multiarch` is an optional parameter for multiarch support.
       - `--no-test-load` is an argument which suppresses the testing phase.
    
    For example, use the following code to install the Windows' version 0.6.1.1 of the R binary with multiarch support:
    ```no-highlight
    install.packages('devtools')
    devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.20/catboost-R-Windows-0.20.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
    ```
    
    {% note warning %}
    
    Do not forget to change the package version to the required one. The example above illustrates the installation of the version 0.6.1.1.
    
    To install another version, let's say 0.16.5, change the code to the following:
    ```
    install.packages('devtools')
    devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.16.5/catboost-R-Windows-0.16.5.tgz', INSTALL_opts = c("--no-multiarch", "--no-test-load"))
    ```
    
    {% endnote %}
    
1. Run the resulting script file.

