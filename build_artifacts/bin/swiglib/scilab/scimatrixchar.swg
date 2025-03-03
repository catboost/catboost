/*
 * C-type: char*
 * Scilab type: string matrix
 */

%include <scichar.swg>

// in (char **IN, int IN_ROWCOUNT, int IN_COLCOUNT)

%typemap(in, noblock=1, fragment="SWIG_SciString_AsCharPtrArrayAndSize") (char **IN, int IN_ROWCOUNT, int IN_COLCOUNT)
{
  if (SWIG_SciString_AsCharPtrArrayAndSize(pvApiCtx, $input, &$2, &$3, &$1, fname) != SWIG_OK) {
    return SWIG_ERROR;
  }
}

// in (int IN_ROWCOUNT, int IN_COLCOUNT, char **IN)

%typemap(in, noblock=1, fragment="SWIG_SciString_AsCharPtrArrayAndSize") (int IN_ROWCOUNT, int IN_COLCOUNT, char **IN)
{
  if (SWIG_SciString_AsCharPtrArrayAndSize(pvApiCtx, $input, &$1, &$2, &$3, fname) != SWIG_OK) {
    return SWIG_ERROR;
  }
}

// in (char **IN, int IN_SIZE)

%typemap(in, noblock=1, fragment="SWIG_SciString_AsCharPtrArrayAndSize") (char **IN, int IN_SIZE) (int rowCount, int colCount)
{
  if (SWIG_SciString_AsCharPtrArrayAndSize(pvApiCtx, $input, &rowCount, &colCount, &$1, fname) == SWIG_OK) {
    $2 = rowCount * colCount;
  }
  else {
    return SWIG_ERROR;
  }
}

// in (int IN_SIZE, char **IN)

%typemap(in, noblock=1, fragment="SWIG_SciString_AsCharPtrArrayAndSize") (int IN_SIZE, char **IN) (int rowCount, int colCount)
{
  if (SWIG_SciString_AsCharPtrArrayAndSize(pvApiCtx, $input, &rowCount, &colCount, &$2, fname) == SWIG_OK) {
    $1 = rowCount * colCount;
  }
  else {
    return SWIG_ERROR;
  }
}

// out (char ***OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)

%typemap(in, noblock=1, numinputs=0) (char ***OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
}

%typemap(arginit, noblock=1) (char ***OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
  $1 = (char***) malloc(sizeof(char**));
  $2 = (int*) malloc(sizeof(int));
  $3 = (int*) malloc(sizeof(int));
}

%typemap(argout, noblock=1, fragment="SWIG_SciString_FromCharPtrArrayAndSize") (char ***OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
  if (SWIG_SciString_FromCharPtrArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), *$2, *$3, *$1) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

%typemap(freearg, noblock=1) (char ***OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
  {
    int i;
    for (i = 0; i < (*$2) * (*$3); i++)
      free((*$1)[i]);
  }
  free(*$1);
  free($1);
  free($2);
  free($3);
}

// out (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, char ***OUT)

%typemap(in, noblock=1, numinputs=0) (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, char ***OUT)
{
}

%typemap(arginit, noblock=1) (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, char ***OUT)
{
  $1 = (char***) malloc(sizeof(char**));
  $2 = (int*) malloc(sizeof(int));
  $3 = (int**) malloc(sizeof(int*));
}

%typemap(argout, noblock=1, fragment="SWIG_SciString_FromCharPtrArrayAndSize") (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, char ***OUT)
{
  if (SWIG_SciString_FromCharPtrArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), *$1, *$2, *$3) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

%typemap(freearg, noblock=1) (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, char ***OUT)
{
  free($1);
  free($2);
  {
    int i;
    for (i = 0; i < (*$1) * (*$2); i++)
      free((*$3)[i]);
  }
  free(*$3);
  free($3);
}


// out (char ***OUT, int *OUT_SIZE)

%typemap(in, noblock=1, numinputs=0) (char ***OUT, int *OUT_SIZE)
{
}

%typemap(arginit, noblock=1) (char ***OUT, int *OUT_SIZE)
{
  $1 = (char***) malloc(sizeof(char**));
  $2 = (int*) malloc(sizeof(int));
}

%typemap(freearg, noblock=1) (char ***OUT, int *OUT_SIZE)
{
  {
    int i;
    for (i = 0; i < *$2; i++)
      free((*$1)[i]);
  }
  free(*$1);
  free($1);
  free($2);
}

%typemap(argout, noblock=1, fragment="SWIG_SciString_FromCharPtrArrayAndSize") (char ***OUT, int *OUT_SIZE)
{
  if (SWIG_SciString_FromCharPtrArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, *$2, *$1) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

// in (int IN_SIZE, char **IN)

%typemap(in, noblock=1, fragment="SWIG_SciString_AsCharPtrArrayAndSize") (int IN_SIZE, char **IN)
{
  if (SWIG_SciString_AsCharPtrArrayAndSize(pvApiCtx, $input, 1, &$1, &$2, fname) != SWIG_OK) {
    return SWIG_ERROR;
  }
}

// out (int *OUT_SIZE, char ***OUT)

%typemap(in, noblock=1, numinputs=0) (int *OUT_SIZE, char ***OUT)
{
}

%typemap(arginit, noblock=1) (int *OUT_SIZE, char ***OUT)
{
  $1 = (int*) malloc(sizeof(int));
  $2 = (char***) malloc(sizeof(char**));
}

%typemap(argout, noblock=1, fragment="SWIG_SciString_FromCharPtrArrayAndSize") (int *OUT_SIZE, char ***OUT)
{
  if (SWIG_SciString_FromCharPtrArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, *$1, *$2) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

%typemap(freearg, noblock=1) (int *OUT_SIZE, char ***OUT)
{
  free($1);
  {
    int i;
    for (i = 0; i < *$1; i++)
      free((*$2)[i]);
  }
  free(*$2);
  free($2);
}

