#### Troubleshooting {#troubleshooting}

{% cut "Installation error" %}

If you use script for installation, try to add the `repos` parameter to the `install.packages('devtools')` command.
Select any repository from the [list](https://cran.r-project.org/mirrors.html).

Example: `install.packages("devtools", repos="http://cran.us.r-project.org")`.

{% endcut %}

{% cut "ERROR: some hard-coded temporary paths could not be fixed" %}

This can be mitigated by adding "--no-staged-install" option to `INSTALL_opts` parameter of `remotes::install_url` call

{% endcut %}
