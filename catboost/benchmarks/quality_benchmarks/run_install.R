method="libcurl"
repos=(c("https://cran.csiro.au"))

install.packages('KScorrect', method=method, repos=repos)
install.packages('reshape2', method=method, repos=repos)

#h2o installation
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods", method=method, repos=repos) }
if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod", method=method, repos=repos) }
if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats", method=method, repos=repos) }
if (! ("graphics" %in% rownames(installed.packages()))) { install.packages("graphics", method=method, repos=repos) }
if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl", method=method, repos=repos) }
if (! ("jsonlite" %in% rownames(installed.packages()))) { install.packages("jsonlite", method=method, repos=repos) }
if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools", method=method, repos=repos) }
if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils", method=method, repos=repos) }

install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/6/R")), method=method)




