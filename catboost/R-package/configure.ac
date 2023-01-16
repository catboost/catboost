# autoconf/configure script for catboost

##### INIT ####################################################################

AC_INIT(catboost,
        m4_esyscmd([cat DESCRIPTION | grep Version: | cut -d " " -f 2 | tr -d '\n']),
        [https://github.com/catboost/catboost/issues])

##### OPTIONS #################################################################

# shell variables/arguments accepted:
# R_HOME - R's home dir
AC_ARG_VAR(R_HOME, [Override the default directory with R installation,
   e.g., /usr/lib64/R, where /usr/lib64 is in the search path.
   Note that $R_HOME/bin/R should point to the R executable.])

AC_ARG_VAR(CATBOOST_DYNLIB, [Optional path to already downloaded or built
   CatBoost dynamic library; either an absolute path or a path relative to
   <package-source-dir>/src.])

##### CHECK FOR R #############################################################

# this is inspired by the "Writing R Extensions" manual:
# determine R_HOME directory...
AC_MSG_CHECKING(for R_HOME)
if test -z "${R_HOME}"; then
   R_HOME=`R RHOME` # set R_HOME if it has not been set already
fi
if test -z "${R_HOME}"; then
   AC_MSG_RESULT(no)
   echo "*** Could not determine R_HOME. Did you install R properly?"
   exit 1
fi
AC_MSG_RESULT($R_HOME)
AC_SUBST(R_HOME)

# ...and then R_PATH
AC_MSG_CHECKING(for R)
R_PATH="${R_HOME}/bin/R" # see "Writing R Extensions"
if test ! -e "${R_PATH}"; then
    # if it determined by calling `R RHOME`, then this is likely a bug
    AC_MSG_RESULT(no)
    echo "*** Could not find R at R_HOME/bin/R, i.e., ${R_HOME}/bin/R"
    exit 1
fi
AC_MSG_RESULT($R_PATH)


##### Get CatBoost dyn lib #####################################################
CATBOOST_FOUND=0
OS_TYPE=$(uname)

# set dynlib suffix
if [[[ $OS_TYPE == 'Darwin' ]]]; then
   DYNLIB_SUFFIX='dylib'
   echo "*** creating .so -> .dylib symlink for working with different R installations"
   ln -s 'libcatboostr.dylib' 'src/libcatboostr.so'
elif [[[ $OS_TYPE == 'Linux' ]]]; then
   DYNLIB_SUFFIX='so'
else
   echo "*** Unsupported system type: not Darwin nor Linux."
   exit 1
fi

DYNLIB_PATH="src/libcatboostr.${DYNLIB_SUFFIX}"

# checking for CATBOOST_DYNLIB variable
AC_MSG_CHECKING(for local CATBOOST_DYNLIB)
if test -n "$CATBOOST_DYNLIB"; then
   if test ! -f "$CATBOOST_DYNLIB"; then
      # check relative path to src/
      if test -f "src/$CATBOOST_DYNLIB"; then
         CATBOOST_DYNLIB="src/$CATBOOST_DYNLIB"
      fi
   fi
   if test -f "$CATBOOST_DYNLIB"; then
      AC_MSG_RESULT($CATBOOST_DYNLIB copying into $DYNLIB_PATH)
      cp -f $CATBOOST_DYNLIB $DYNLIB_PATH
      CATBOOST_FOUND=1
   fi
fi

if test $CATBOOST_FOUND = 0; then
   # check if src/libcatboostr.{so/darwin} exists already
   if test -f "$DYNLIB_PATH"; then
      CATBOOST_DYNLIB=$DYNLIB_PATH
      AC_MSG_RESULT($CATBOOST_DYNLIB)
      CATBOOST_FOUND=1
   else
      AC_MSG_RESULT(no)
   fi
fi

if test $CATBOOST_FOUND = 0; then
   # download catboost
   AC_MSG_CHECKING([whether we can fetch CatBoost dynlib])
   "${R_PATH}" --vanilla --slave -e " \
           source(\"R/install.R\"); \
           if (identical(FALSE,catboost_download_dynlib(\"src/\"))) \
               stop(\"Stopping on error\")"
   if test $? -ne 0; then
      echo "*** CatBoost dynamic library download failed. stopping."
      exit 1
   fi
fi

AC_SUBST(CATBOOST_DYNLIB)
AC_OUTPUT
