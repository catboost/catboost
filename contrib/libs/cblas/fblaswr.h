real 
sdot_(integer* N, 
      real* X, integer* incX, 
      real* Y, integer* incY);

doublereal
ddot_(integer* N, 
      doublereal* X, integer* incX, 
      doublereal* Y, integer* incY);

void 
cdotu_(complex* retval,
       integer* N, 
       complex* X, integer* incX, 
       complex* Y, integer* incY);

void
cdotc_(complex* retval,
       integer* N, 
       complex* X, integer* incX, 
       complex* Y, integer* incY);

void
zdotu_(doublecomplex* retval,
       integer* N, 
       doublecomplex* X, integer* incX, 
       doublecomplex* Y, integer* incY);

void
zdotc_(doublecomplex* retval,
       integer* N, 
       doublecomplex* X, integer* incX, 
       doublecomplex* Y, integer* incY);

real 
snrm2_(integer* N, 
       real* X, integer* incX);

real
sasum_(integer* N, 
       real* X, integer* incX);

doublereal
dnrm2_(integer* N, 
       doublereal* X, integer* incX);

doublereal
dasum_(integer* N, 
       doublereal* X, integer* incX);

real 
scnrm2_(integer* N, 
        complex* X, integer* incX);

real
scasum_(integer* N, 
        complex* X, integer* incX);

doublereal 
dznrm2_(integer* N, 
        doublecomplex* X, integer* incX);

doublereal
dzasum_(integer* N, 
        doublecomplex* X, integer* incX);

integer
isamax_(integer* N,
        real* X, integer* incX);

integer
idamax_(integer* N,
        doublereal* X, integer* incX);

integer
icamax_(integer* N,
        complex* X, integer* incX);

integer
izamax_(integer* N,
        doublecomplex* X, integer* incX);

int
sswap_(integer* N,
       real* X, integer* incX,
       real* Y, integer* incY);

int
scopy_(integer* N,
       real* X, integer* incX,
       real* Y, integer* incY);

int
saxpy_(integer* N,
       real* alpha,
       real* X, integer* incX,
       real* Y, integer* incY);

int
dswap_(integer* N,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY);

int
dcopy_(integer* N,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY);

int
daxpy_(integer* N,
       doublereal* alpha,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY);

int
cswap_(integer* N,
       complex* X, integer* incX,
       complex* Y, integer* incY);

int
ccopy_(integer* N,
       complex* X, integer* incX,
       complex* Y, integer* incY);

int
caxpy_(integer* N,
      complex* alpha,
      complex* X, integer* incX,
      complex* Y, integer* incY);

int
zswap_(integer* N,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY);

int
zcopy_(integer* N,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY);

int
zaxpy_(integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY);

int
srotg_(real* a, real* b, real* c, real* s);

int
srot_(integer* N,
      real* X, integer* incX,
      real* Y, integer* incY,
      real* c, real* s);

int
crotg_(complex* a, complex* b, complex* c, complex* s);

int
drotg_(doublereal* a, doublereal* b, doublereal* c, doublereal* s);

int
drot_(integer* N,
      doublereal* X, integer* incX,
      doublereal* Y, integer* incY,
      doublereal* c, doublereal* s);

int
zrotg_(doublecomplex* a, doublecomplex* b, doublecomplex* c, doublecomplex* s);

int
sscal_(integer* N,
       real* alpha,
       real* X, integer* incX);

int
dscal_(integer* N,
       doublereal* alpha,
       doublereal* X, integer* incX);

int
cscal_(integer* N,
       complex* alpha,
       complex* X, integer* incX);

int
zscal_(integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX);

int
csscal_(integer* N,
        real* alpha,
        complex* X, integer* incX);

int
zdscal_(integer* N,
        doublereal* alpha,
        doublecomplex* X, integer* incX);

int
sgemv_(char* trans, integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int
sgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       real *alpha, 
       real *A, integer *lda, 
       real *X, integer *incX, 
       real *beta, 
       real *Y, integer *incY);

int 
strmv_(char* uplo, char *trans, char* diag, integer *N,  
       real *A, integer *lda, 
       real *X, integer *incX);

int
stbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       real* A, integer* lda,
       real* X, integer* incX);

int
stpmv_(char* uplo, char* trans, char* diag, integer* N, 
       real* Ap, 
       real* X, integer* incX);

int
strsv_(char* uplo, char* trans, char* diag, integer* N,
       real* A, integer* lda,
       real* X, integer* incX);

int
stbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       real* A, integer* lda, 
       real* X, integer* incX);

int
stpsv_(char* uplo, char* trans, char* diag, integer* N, 
       real* Ap, 
       real* X, integer* incX);

int
dgemv_(char* trans, integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int 
dgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       doublereal *alpha, 
       doublereal *A, integer *lda, 
       doublereal *X, integer *incX, 
       doublereal *beta, 
       doublereal *Y, integer *incY);

int 
dtrmv_(char* uplo, char *trans, char* diag, integer *N,  
       doublereal *A, integer *lda, 
       doublereal *X, integer *incX);

int
dtbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX);

int
dtpmv_(char* uplo, char* trans, char* diag, integer* N, 
       doublereal* Ap, 
       doublereal* X, integer* incX);

int
dtrsv_(char* uplo, char* trans, char* diag, integer* N,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX);

int
dtbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublereal* A, integer* lda, 
       doublereal* X, integer* incX);

int
dtpsv_(char* uplo, char* trans, char* diag, integer* N, 
       doublereal* Ap, 
       doublereal* X, integer* incX);

int
cgemv_(char* trans, integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int 
cgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       complex *alpha, 
       complex *A, integer *lda, 
       complex *X, integer *incX, 
       complex *beta, 
       complex *Y, integer *incY);

int 
ctrmv_(char* uplo, char *trans, char* diag, integer *N,  
       complex *A, integer *lda, 
       complex *X, integer *incX);

int
ctbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       complex* A, integer* lda,
       complex* X, integer* incX);

int
ctpmv_(char* uplo, char* trans, char* diag, integer* N, 
       complex* Ap, 
       complex* X, integer* incX);

int
ctrsv_(char* uplo, char* trans, char* diag, integer* N,
       complex* A, integer* lda,
       complex* X, integer* incX);

int
ctbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       complex* A, integer* lda, 
       complex* X, integer* incX);

int
ctpsv_(char* uplo, char* trans, char* diag, integer* N, 
       complex* Ap, 
       complex* X, integer* incX);

int
zgemv_(char* trans, integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int 
zgbmv_(char *trans, integer *M, integer *N, integer *KL, integer *KU, 
       doublecomplex *alpha, 
       doublecomplex *A, integer *lda, 
       doublecomplex *X, integer *incX, 
       doublecomplex *beta, 
       doublecomplex *Y, integer *incY);

int 
ztrmv_(char* uplo, char *trans, char* diag, integer *N,  
       doublecomplex *A, integer *lda, 
       doublecomplex *X, integer *incX);

int
ztbmv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX);

 void  
ztpmv_(char* uplo, char* trans, char* diag, integer* N, 
      doublecomplex* Ap, 
      doublecomplex* X, integer* incX);

int
ztrsv_(char* uplo, char* trans, char* diag, integer* N,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX);

int
ztbsv_(char* uplo, char* trans, char* diag, integer* N, integer* K,
       doublecomplex* A, integer* lda, 
       doublecomplex* X, integer* incX);

int
ztpsv_(char* uplo, char* trans, char* diag, integer* N, 
       doublecomplex* Ap, 
       doublecomplex* X, integer* incX);

int
ssymv_(char* uplo, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int 
ssbmv_(char* uplo, integer* N, integer* K,
       real* alpha,
       real* A, integer* lda,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int
sspmv_(char* uplo, integer* N,
       real* alpha,
       real* Ap,
       real* X, integer* incX,
       real* beta,
       real* Y, integer* incY);

int
sger_(integer* M, integer* N,
      real* alpha,
      real* X, integer* incX,
      real* Y, integer* incY,
      real* A, integer* lda);

int
ssyr_(char* uplo, integer* N,
      real* alpha,
      real* X, integer* incX,
      real* A, integer* lda);

int
sspr_(char* uplo, integer* N,
      real* alpha,
      real* X, integer* incX,
      real* Ap);

int
ssyr2_(char* uplo, integer* N,
       real* alpha,
       real* X, integer* incX,
       real* Y, integer* incY,
       real* A, integer* lda);

int
sspr2_(char* uplo, integer* N,
       real* alpha, 
       real* X, integer* incX,
       real* Y, integer* incY,
       real* A);

int
dsymv_(char* uplo, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int 
dsbmv_(char* uplo, integer* N, integer* K,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int
dspmv_(char* uplo, integer* N,
       doublereal* alpha,
       doublereal* Ap,
       doublereal* X, integer* incX,
       doublereal* beta,
       doublereal* Y, integer* incY);

int
dger_(integer* M, integer* N,
      doublereal* alpha,
      doublereal* X, integer* incX,
      doublereal* Y, integer* incY,
      doublereal* A, integer* lda);

int
dsyr_(char* uplo, integer* N,
      doublereal* alpha,
      doublereal* X, integer* incX,
      doublereal* A, integer* lda);

int
dspr_(char* uplo, integer* N,
      doublereal* alpha,
      doublereal* X, integer* incX,
      doublereal* Ap);

int
dsyr2_(char* uplo, integer* N,
       doublereal* alpha,
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY,
       doublereal* A, integer* lda);

int
dspr2_(char* uplo, integer* N,
       doublereal* alpha, 
       doublereal* X, integer* incX,
       doublereal* Y, integer* incY,
       doublereal* A);

int
chemv_(char* uplo, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int
chbmv_(char* uplo, integer* N, integer* K,
       complex* alpha,
       complex* A, integer* lda,
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int
chpmv_(char* uplo, integer* N, 
       complex* alpha,
       complex* Ap, 
       complex* X, integer* incX,
       complex* beta,
       complex* Y, integer* incY);

int
cgeru_(integer* M, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* A, integer* lda);

int
cgerc_(integer* M, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* A, integer* lda);

int
cher_(char* uplo, integer* N,
      real* alpha,
      complex* X, integer* incX,
      complex* A, integer* lda);

int
chpr_(char* uplo, integer* N,
      real* alpha,
      complex* X, integer* incX,
      complex* Ap);

int
cher2_(char* uplo, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* A, integer* lda);

int
chpr2_(char* uplo, integer* N,
       complex* alpha,
       complex* X, integer* incX,
       complex* Y, integer* incY,
       complex* Ap);

int
zhemv_(char* uplo, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int
zhbmv_(char* uplo, integer* N, integer* K,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int
zhpmv_(char* uplo, integer* N, 
       doublecomplex* alpha,
       doublecomplex* Ap, 
       doublecomplex* X, integer* incX,
       doublecomplex* beta,
       doublecomplex* Y, integer* incY);

int
zgeru_(integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* A, integer* lda);

int
zgerc_(integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* A, integer* lda);

int
zher_(char* uplo, integer* N,
      doublereal* alpha,
      doublecomplex* X, integer* incX,
      doublecomplex* A, integer* lda);

int
zhpr_(char* uplo, integer* N,
      doublereal* alpha,
      doublecomplex* X, integer* incX,
      doublecomplex* Ap);

int
zher2_(char* uplo, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* A, integer* lda);

int
zhpr2_(char* uplo, integer* N,
       doublecomplex* alpha,
       doublecomplex* X, integer* incX,
       doublecomplex* Y, integer* incY,
       doublecomplex* Ap);

int
sgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb,
       real* beta,
       real* C, integer* ldc);

int
ssymm_(char* side, char* uplo, integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb,
       real* beta,
       real* C, integer* ldc);

int
ssyrk_(char* uplo, char* trans, integer* N, integer* K,
       real* alpha,
       real* A, integer* lda,
       real* beta,
       real* C, integer* ldc);

int
ssyr2k_(char* uplo, char* trans, integer* N, integer* K,
        real* alpha,
        real* A, integer* lda,
        real* B, integer* ldb,
        real* beta,
        real* C, integer* ldc);

int
strmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb);

int 
strsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       real* alpha,
       real* A, integer* lda,
       real* B, integer* ldb);

int
dgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb,
       doublereal* beta,
       doublereal* C, integer* ldc);

int
dsymm_(char* side, char* uplo, integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb,
       doublereal* beta,
       doublereal* C, integer* ldc);

int
dsyrk_(char* uplo, char* trans, integer* N, integer* K,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* beta,
       doublereal* C, integer* ldc);

int
dsyr2k_(char* uplo, char* trans, integer* N, integer* K,
        doublereal* alpha,
        doublereal* A, integer* lda,
        doublereal* B, integer* ldb,
        doublereal* beta,
        doublereal* C, integer* ldc);

int
dtrmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb);

int 
dtrsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       doublereal* alpha,
       doublereal* A, integer* lda,
       doublereal* B, integer* ldb);

int
cgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb,
       complex* beta,
       complex* C, integer* ldc);

int
csymm_(char* side, char* uplo, integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb,
       complex* beta,
       complex* C, integer* ldc);

int
csyrk_(char* uplo, char* trans, integer* N, integer* K,
       complex* alpha,
       complex* A, integer* lda,
       complex* beta,
       complex* C, integer* ldc);

int
csyr2k_(char* uplo, char* trans, integer* N, integer* K,
        complex* alpha,
        complex* A, integer* lda,
        complex* B, integer* ldb,
        complex* beta,
        complex* C, integer* ldc);

int
ctrmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb);

int 
ctrsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb);

int
zgemm_(char* transA, char* transB, integer* M, integer* N, integer* K,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
zsymm_(char* side, char* uplo, integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
zsyrk_(char* uplo, char* trans, integer* N, integer* K,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
zsyr2k_(char* uplo, char* trans, integer* N, integer* K,
        doublecomplex* alpha,
        doublecomplex* A, integer* lda,
        doublecomplex* B, integer* ldb,
        doublecomplex* beta,
        doublecomplex* C, integer* ldc);

int
ztrmm_(char* side, char* uplo, char* trans, char* diag, 
       integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb);

int 
ztrsm_(char* side, char* uplo, char* trans, char* diag,
       integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb);

int
chemm_(char* side, char* uplo, integer* M, integer* N,
       complex* alpha,
       complex* A, integer* lda,
       complex* B, integer* ldb,
       complex* beta,
       complex* C, integer* ldc);

int
cherk_(char* uplo, char* trans, integer* N, integer* K,
       real* alpha,
       complex* A, integer* lda,
       real* beta,
       complex* C, integer* ldc);

int
cher2k_(char* uplo, char* trans, integer* N, integer* K,
        complex* alpha,
        complex* A, integer* lda,
        complex* B, integer* ldb,
        real* beta,
        complex* C, integer* ldc);

int
zhemm_(char* side, char* uplo, integer* M, integer* N,
       doublecomplex* alpha,
       doublecomplex* A, integer* lda,
       doublecomplex* B, integer* ldb,
       doublecomplex* beta,
       doublecomplex* C, integer* ldc);

int
zherk_(char* uplo, char* trans, integer* N, integer* K,
       doublereal* alpha,
       doublecomplex* A, integer* lda,
       doublereal* beta,
       doublecomplex* C, integer* ldc);

int
zher2k_(char* uplo, char* trans, integer* N, integer* K,
        doublecomplex* alpha,
        doublecomplex* A, integer* lda,
        doublecomplex* B, integer* ldb,
        doublereal* beta,
        doublecomplex* C, integer* ldc);
