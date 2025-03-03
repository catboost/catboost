/*
 * C-type: int array
 * Scilab type: 32-bit integer matrix
 */

%include <sciint.swg>

// in (int *IN, int IN_ROWCOUNT, int IN_COLCOUNT)

%typemap(in, noblock=1, fragment="SWIG_SciDoubleOrInt32_AsIntArrayAndSize") (int *IN, int IN_ROWCOUNT, int IN_COLCOUNT)
{
  if (SWIG_SciDoubleOrInt32_AsIntArrayAndSize(pvApiCtx, $input, &$2, &$3, &$1, fname) != SWIG_OK) {
    return SWIG_ERROR;
  }
}


// in (int IN_ROWCOUNT, int IN_COLCOUNT, int *IN)

%typemap(in, noblock=1, fragment="SWIG_SciDoubleOrInt32_AsIntArrayAndSize") (int IN_ROWCOUNT, int IN_COLCOUNT, int *IN)
{
  if (SWIG_SciDoubleOrInt32_AsIntArrayAndSize(pvApiCtx, $input, &$1, &$2, &$3, fname) != SWIG_OK) {
    return SWIG_ERROR;
  }
}


// in (int *IN, int IN_SIZE)

%typemap(in, noblock=1, fragment="SWIG_SciDoubleOrInt32_AsIntArrayAndSize") (int *IN, int IN_SIZE) (int rowCount, int colCount)
{
  if (SWIG_SciDoubleOrInt32_AsIntArrayAndSize(pvApiCtx, $input, &rowCount, &colCount, &$1, fname) == SWIG_OK) {
    $2 = rowCount * colCount;
  }
  else {
    return SWIG_ERROR;
  }
}


// in (int IN_SIZE, int *IN)

%typemap(in, noblock=1, fragment="SWIG_SciDoubleOrInt32_AsIntArrayAndSize") (int IN_SIZE, int *IN) (int rowCount, int colCount)
{
  if (SWIG_SciDoubleOrInt32_AsIntArrayAndSize(pvApiCtx, $input, &rowCount, &colCount, &$2, fname) == SWIG_OK) {
	$1 = rowCount * colCount;
  }
  else {
	return SWIG_ERROR;
  }
}

// out (int **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)

%typemap(in, noblock=1, numinputs=0) (int **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
}

%typemap(arginit, noblock=1) (int **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
  $1 = (int**) malloc(sizeof(int*));
  $2 = (int*) malloc(sizeof(int));
  $3 = (int*) malloc(sizeof(int));
}

%typemap(argout, noblock=1, fragment="SWIG_SciDouble_FromIntArrayAndSize") (int **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
  if (SWIG_SciDouble_FromIntArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), *$2, *$3, *$1) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

%typemap(freearg, noblock=1) (int **OUT, int *OUT_ROWCOUNT, int *OUT_COLCOUNT)
{
  free(*$1);
  free($1);
  free($2);
  free($3);
}


// out (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, int **OUT)

%typemap(in, noblock=1, numinputs=0) (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, int **OUT)
{
}

%typemap(arginit, noblock=1) (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, int **OUT)
{
  $1 = (int*) malloc(sizeof(int));
  $2 = (int*) malloc(sizeof(int));
  $3 = (int**) malloc(sizeof(int*));
}

%typemap(argout, noblock=1, fragment="SWIG_SciDouble_FromIntArrayAndSize") (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, int **OUT)
{
  if (SWIG_SciDouble_FromIntArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), *$1, *$2, *$3) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

%typemap(freearg, noblock=1) (int *OUT_ROWCOUNT, int *OUT_COLCOUNT, int **OUT)
{
  free($1);
  free($2);
  free(*$3);
  free($3);
}


// out (int **OUT, int *OUT_SIZE)

%typemap(in, noblock=1, numinputs=0) (int **OUT, int *OUT_SIZE)
{
}

%typemap(arginit) (int **OUT, int *OUT_SIZE)
{
  $1 = (int**) malloc(sizeof(int*));
  $2 = (int*) malloc(sizeof(int));
}

%typemap(argout, noblock=1, fragment="SWIG_SciDouble_FromIntArrayAndSize") (int **OUT, int *OUT_SIZE)
{
  if (SWIG_SciDouble_FromIntArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, *$2, *$1) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

%typemap(freearg, noblock=1) (int **OUT, int *OUT_SIZE)
{
  free(*$1);
  free($1);
  free($2);
}


// out (int *OUT_SIZE, int **OUT)

%typemap(in, noblock=1, numinputs=0) (int *OUT_SIZE, int **OUT)
{
}

%typemap(arginit, noblock=1) (int *OUT_SIZE, int **OUT)
{
  $1 = (int*) malloc(sizeof(int));
  $2 = (int**) malloc(sizeof(int*));
}

%typemap(argout, noblock=1, fragment="SWIG_SciDouble_FromIntArrayAndSize") (int *OUT_SIZE, int **OUT)
{
  if (SWIG_SciDouble_FromIntArrayAndSize(pvApiCtx, SWIG_Scilab_GetOutputPosition(), 1, *$1, *$2) == SWIG_OK) {
    SWIG_Scilab_SetOutput(pvApiCtx, SWIG_NbInputArgument(pvApiCtx) + SWIG_Scilab_GetOutputPosition());
  }
  else {
    return SWIG_ERROR;
  }
}

%typemap(freearg, noblock=1) (int *IN_SIZE, int **OUT)
{
  free($1);
  free(*$2);
  free($2);
}

