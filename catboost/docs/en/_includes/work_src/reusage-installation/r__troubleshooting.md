#### Troubleshooting {#troubleshooting}

{% cut "Installation error" %}

If you use script for installation, try to add the `repos` parameter to the `install.packages('devtools')` command.
Select any repository from the [list](https://cran.r-project.org/mirrors.html).

Example: `install.packages("devtools", repos="http://cran.us.r-project.org")`.

{% endcut %}
