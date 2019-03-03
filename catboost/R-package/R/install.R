catboost_download_dynlib <- function(dest_dir) {
  ver <- paste0("v", read.dcf("DESCRIPTION")[1, "Version"])

  base_url <- "https://github.com/catboost/catboost/releases/download"
  if (.Platform$OS.type == "windows") {
    src_dynlib <- "libcatboostr.dll"
    dst_dynlib <- src_dynlib
  } else if (grepl("darwin", R.version$os)) {
    src_dynlib <- "libcatboostr-darwin.so"
    dst_dynlib <- "libcatboostr.so"
  } else if (.Platform$OS.type == "unix") {
    src_dynlib <- "libcatboostr-linux.so"
    dst_dynlib <- "libcatboostr.so"
  } else {
    return(FALSE)
  }
  url <- paste(base_url, ver, src_dynlib, sep = "/")

  message(sprintf("downloading CatBoost (%s - %s)", dst_dynlib, ver))
  status <- tryCatch({
    if (!dir.exists(dest_dir)) {
      dir.create(dest_dir, showWarnings = FALSE, recursive = TRUE)
    }
    dest_fpath <- file.path(dest_dir, dst_dynlib)
    suppressWarnings(file.remove(dest_fpath))
    if (download.file(url, dest_fpath, mode = "wb") == 0) {
      message("CatBoost fetch successful")
      TRUE
    } else {
      message("CatBoost download error")
      FALSE
    }
  },
  error = function(e) {
    message(as.character(e))
    return(FALSE)
  })

  return(status)
}
