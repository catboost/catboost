"""Download the UNU.RAN library and clean it for use in SciPy."""

import os
import re
import argparse
import gzip
import logging
import pkg_resources
import pathlib
import platform
import urllib.request
import shutil
import tarfile
import zipfile
import tempfile
import datetime
from typing import Tuple, List

logging.basicConfig()


def _download_unuran(version: str, logger: logging.Logger) -> None:
    # base is where this script is located
    base = pathlib.Path(__file__).parent
    UNURAN_VERSION = pkg_resources.parse_version(version).base_version
    archive_name = f"unuran-{UNURAN_VERSION}"
    suffix = "tar.gz"

    # some version need zip files for windows
    # see http://statmath.wu.ac.at/src/ for a list of such files
    winversions = ["0.5.0", "0.6.0", "1.0.1", "1.1.0", "1.2.0"]

    # 32-bit windows has different file name
    # see http://statmath.wu.ac.at/src/ for a list of such files
    is_win32: bool = platform.system() == "Windows" and platform.architecture()[0] == "32bit"

    # replace suffix for windows if the version is in winversion.
    if platform.system() == "Windows" and UNURAN_VERSION in winversions:
        archive_name += "-win"
        if is_win32 and UNURAN_VERSION in winversions[2:]:
            archive_name += "32"
        suffix = "zip"

    # download url
    url = f"http://statmath.wu.ac.at/src/{archive_name}.{suffix}"

    # Start download
    logger.info(f" Downloading UNU.RAN version {UNURAN_VERSION} from {url}")
    start = datetime.datetime.now()
    with urllib.request.urlopen(url) as response:
        # Uncompress .tar.gz files before extracting.
        if suffix == "tar.gz":
            with gzip.GzipFile(fileobj=response) as uncompressed, tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as ntf:
                logger.info(f" Saving UNU.RAN tarball to {ntf.name}")
                shutil.copyfileobj(uncompressed, ntf)
                ntf.flush()

        # uncompressing finished
        logger.info(f" Finished downloading (and uncompressing) in {datetime.datetime.now() - start}")

        # Start extraction
        logger.info(" Starting to extract")
        start = datetime.datetime.now()
        with tempfile.TemporaryDirectory() as tmpdir:
            # temporary destination for extracted files.
            dst = pathlib.Path(tmpdir)

            # use tarfile for .tar tarball
            if suffix == "tar.gz":
                try:
                    with tarfile.open(ntf.name, "r") as tar:
                        def is_within_directory(directory, target):
                            
                            abs_directory = os.path.abspath(directory)
                            abs_target = os.path.abspath(target)
                        
                            prefix = os.path.commonprefix([abs_directory, abs_target])
                            
                            return prefix == abs_directory
                        
                        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                        
                            for member in tar.getmembers():
                                member_path = os.path.join(path, member.name)
                                if not is_within_directory(path, member_path):
                                    raise Exception("Attempted Path Traversal in Tar File")
                        
                            tar.extractall(path, members, numeric_owner=numeric_owner) 
                            
                        
                        safe_extract(tar, path=dst)
                finally:
                    # We want to save the tar file as a temporary file and simulataneously
                    # extract it, meaning it will need to be opened in a context manager
                    # multiple times. While Linux can handle nested context managers
                    # using the same file handle, Windows cannot.  So we have to mark the
                    # temporary file "ntf" as delete=False, close its context manager, and
                    # then ensure cleanup happens in this "finally" statement
                    ntf.close()
            # handle zip files on windows
            else:
                with zipfile.ZipFile(response, "r") as zip:
                    zip.extractall(path=dst)
            logger.info(f" Finished extracting to {dst / archive_name} in {datetime.datetime.now() - start}")

            # if "unuran" directory already exists, remove it.
            if (base / "unuran").exists():
                shutil.rmtree(base / "unuran")
            shutil.move(dst / archive_name, base / "unuran")

        # These files are moved to the rood directory.
        files_to_move: List[Tuple[str, str]] = [
            ("README", "UNURAN_README.txt"),
            ("README.win32", "UNURAN_README_win32.txt"),
            ("ChangeLog", "UNURAN_ChangeLog"),
            ("AUTHORS", "UNURAN_AUTHORS"),
            ("THANKS", "UNURAN_THANKS"),
        ]
        for file_to_move in files_to_move:
            if (base / "unuran" / file_to_move[0]).exists():
                shutil.move(base / "unuran" / file_to_move[0], base / file_to_move[1])

        # Unwanted directories.
        dirs_to_remove = ["src/uniform", "autoconf", "tests", "doc", "examples", "experiments", "scripts"]
        for dir_to_remove in dirs_to_remove:
            if (base / "unuran" / dir_to_remove).exists():
                shutil.rmtree(base / "unuran" / dir_to_remove)

        # Unwanted files.
        files_to_remove = ["src/unuran.h.in", "acinclude.m4", "aclocal.m4", "autogen.sh", "configure",
                           "COPYING", "INSTALL", "NEWS", "UPGRADE", "src/specfunct/log1p.c"]
        for file_to_remove in files_to_remove:
            if (base / "unuran" / file_to_remove).exists():
                os.remove(base / "unuran" / file_to_remove)


def _clean_makefiles(logger: logging.Logger) -> None:
    # Remove unwanted Makefiles.
    logger.info(" Removing Makefiles")
    base = pathlib.Path(__file__).parent / "unuran"
    # remove Makefiles from all the directories under unuran.
    dirs = ["./Makefile*", "./*/Makefile*", "./*/*/Makefile*"]
    for dir in dirs:
        for p in base.glob(dir):
            logger.info(f"     Removing {str(p)}")
            p.unlink()
    logger.info(" Complete")


def _clean_deprecated(logger: logging.Logger) -> None:
    # Remove deprecated files
    # This needs to be done in 2 steps:
    #   1. Remove the deprecated files.
    #   2. Remove their declaration from unuran.h
    logger.info(" Removing deprecated files")
    base = pathlib.Path(__file__).parent / "unuran"
    for p in base.glob("./*/*/deprecated*"):
        logger.info(f"     Removing {str(p)}")
        p.unlink()
    with open(base / "src" / "unuran.h", "rb") as f:
        # Some files contain non utf-8 characters. Ignore them.
        content = f.read().decode("utf-8", "ignore")

    # All the declaration must match this regular expression:
    # /* <1> `deprecated_*.h' */
    # ... (declarations)
    # /* end of `deprecated_*.h' */
    # So, we can use re.sub to remove all deprecated declarations from unuran.h
    content = re.sub(r"/\* <1> `deprecated_(.*).h\' \*/(.|\n)*/\* end of `deprecated_(.*).h\' \*/",
                     r"/* Removed `deprecated_\1.h' for use in SciPy */", content)
    with open(base / "src" / "unuran.h", "w") as f:
        f.write(content)
    logger.info(" Complete")


def _ch_to_h(logger: logging.Logger) -> None:
    # Rename .ch files
    # We also need to edit all the sources under that directory
    # to use the .h files instead of .ch files.
    logger.info(" Renaming `.ch` -> `.h`")
    base = pathlib.Path(__file__).parent / "unuran"
    for p in base.glob("./*/*/*.ch"):
        logger.info(f'     Renaming {str(p):30s} -> {str(p.parent / p.name[:-2]) + "h":30s}')

        # `.ch` --> `.h`
        p.rename(p.parent / (p.name[:-2] + "h"))
        all_files = os.listdir(p.parent)

        # edit source files to include .h files instead
        for file in all_files:
            with open(p.parent / file, "rb") as f:
                content = f.read().decode("utf-8", "ignore")
            content = re.sub(r"#include <(.*).ch>", r"#include <\1.h>", content)
            content = re.sub(r'#include "(.*).ch"', r'#include "\1.h"', content)
            with open(p.parent / file, "w") as f:
                f.write(content)
    logger.info(" Complete")


def _replace_urng_default(logger: logging.Logger) -> None:
    # This is messy right now. But I am not sure if there is a good way
    # to do this. We need to modify the urng_default file and some headers
    # in unuran_config.h to remove the default URNG. The approach I have
    # followed is: replace the file `urng_default.c` with a modified version
    # `urng_default_mod.c`, then remove the declarations of default URNG from
    # unuran.h, and finally replace some macros in unuran_config.h to not use
    # the default RNG.
    logger.info(" Replacing URNG API with a modified version for SciPy")

    # Replace the file urng_default.c with urng_default_mod.c
    basefile = pathlib.Path(".") / "urng_default_mod.c"
    repfile = pathlib.Path(__file__).parent / "unuran" / "src" / "urng" / "urng_default.c"
    shutil.copy(basefile, repfile)

    # remove declarations of default URNG from unuran.h
    removed_headers = [r"urng_builtin.h", r"urng_fvoid.h", r"urng_randomshift.h"]
    with open(basefile.parent / "unuran" / "src" / "unuran.h", "rb") as f:
        content = f.read().decode("utf-8", "ignore")
    for header in removed_headers:
        content = re.sub(rf"/\* <1> `{header}\' \*/(.|\n)*/\* end of `{header}\' \*/",
                         rf"/* Removed `{header}' for use in SciPy */", content)
    with open(basefile.parent / "unuran" / "src" / "unuran.h", "w") as f:
        f.write(content)
    with open(basefile.parent / "unuran" / "src" / "unuran_config.h", "rb") as f:
        content = f.read().decode("utf-8", "ignore")

    # replace macros in unuran_config.h to not use the default URNG.
    content = re.sub(
        r"# *define *UNUR_URNG_DEFAULT *\(?unur_urng_builtin\(\)?\)",
        r"#define UNUR_URNG_DEFAULT unur_get_default_urng()",
        content,
    )
    content = re.sub(
        r"# *define *UNUR_URNG_AUX_DEFAULT *\(?unur_urng_builtin_aux\(\)?\)",
        r"#define UNUR_URNG_AUX_DEFAULT unur_get_default_urng_aux()",
        content,
    )
    with open(basefile.parent / "unuran" / "src" / "unuran_config.h", "w") as f:
        f.write(content)
    logger.info(" Complete")


def _remove_misc(logger: logging.Logger):
    logger.info(" Removing miscellaneous files...")
    misc_files = [
        "./*/*/*.pl",
        "./*/*/*.in",
        "./*/*/*.dh"
    ]
    base = pathlib.Path(__file__).parent / "unuran"
    for file in misc_files:
        for p in base.glob(file):
            logger.info(f"     Removing {str(p)}")
            p.unlink()
    logger.info(" Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--unuran-version",
        type=str,
        help="UNU.RAN version to download formatted as [major].[minor].[patch].",
        default="1.8.1",
    )
    parser.add_argument(
        "-v", action="store_true", help="Enable verbose logging.", default=False
    )
    args = parser.parse_args()
    logger = logging.getLogger("get-and-clean-unuran")
    if args.v:
        logger.setLevel(logging.INFO)
    _download_unuran(version=args.unuran_version, logger=logger)
    _clean_makefiles(logger)
    _clean_deprecated(logger)
    _ch_to_h(logger)
    _replace_urng_default(logger)
    _remove_misc(logger)
