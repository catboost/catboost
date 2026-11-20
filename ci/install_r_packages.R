packages <- commandArgs(trailingOnly = TRUE)
stopifnot(length(packages) > 0L)

missing_packages <- function() {
    packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
}

for (attempt in seq_len(3L)) {
    missing <- missing_packages()
    if (!length(missing)) {
        break
    }
    message("Installing R packages (attempt ", attempt, "/3): ", paste(missing, collapse = ", "))
    tryCatch(
        install.packages(missing, Ncpus = parallel::detectCores()),
        error = function(error) message(conditionMessage(error))
    )
    # install.packages can report download failures as warnings and return
    # successfully, so check that the requested namespaces actually load.
    if (!length(missing_packages())) {
        break
    }
    if (attempt < 3L) {
        Sys.sleep(5L)
    }
}

missing <- missing_packages()
if (length(missing)) {
    stop("R packages unavailable after 3 installation attempts: ", paste(missing, collapse = ", "))
}
