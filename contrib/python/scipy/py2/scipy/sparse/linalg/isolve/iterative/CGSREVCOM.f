*  -*- fortran -*-
      SUBROUTINE sCGSREVCOM(N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                     NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the 
*     Solution of Linear Systems: Building Blocks for Iterative 
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra, 
*     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, LDW, ITER, INFO
      real    RESID
      INTEGER            NDX1, NDX2
      real   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      real   X( * ), B( * ), WORK( LDW,* )
*     ..
*
*  Purpose
*  =======
*
*  CGS solves the linear system Ax = b using the
*  Conjugate Gradient Squared iterative method with preconditioning.
*
*  Convergence test: ( norm( b - A*x ) / norm( b ) ) .ls.  TOL.
*  For other measures, see the above reference.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER. 
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
* 
*  B       (input) DOUBLE PRECISION array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess. This is commonly set to
*          the zero vector. The user should be warned that for
*          this particular algorithm, an initial guess close to
*          the actual solution can result in divergence.
*          On exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,7)
*          Workspace for residual, direction vector, etc.
*          Note that vectors PHAT and QHAT, and UHAT and VHAT share 
*          the same workspace.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input) DOUBLE PRECISION
*          On input, the allowable convergence measure for
*          norm( b - A*x ).
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit.
*          .gt.   0: Convergence not achieved. This will be set
*                to the number of iterations performed.
*
*          .ls.   0: Illegal input parameter, or breakdown occurred
*                during iteration.
*
*                Illegal parameter: 
*
*                   -1: matrix dimension N .ls.  0
*                   -2: LDW .ls.  N
*                   -3: Maximum number of iterations ITER .ls. = 0.
*                   -5: Erroneous NDX1/NDX2 in INIT call.
*                   -6: Erroneous RLBL.
*
*                BREAKDOWN: If RHO become smaller than some tolerance,
*                   the program will terminate. Here we check 
*                   against tolerance BREAKTOL.
*
*                   -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                        orthogonal.
*
*  NDX1    (input/output) INTEGER. 
*  NDX2    On entry in INIT call contain indices required by interface
*          level for stopping test.
*          All other times, used as output, to indicate indices into
*          WORK[] for the MATVEC, PSOLVE done by the interface level.
*
*  SCLR1   (output) DOUBLE PRECISION.
*  SCLR2   Used to pass the scalars used in MATVEC. Scalars are reqd because
*          original routines use dgemv.
*
*  IJOB    (input/output) INTEGER. 
*          Used to communicate job code between the two levels.
*
*  BLAS CALLS:    DAXPY, DCOPY, DDOT, DNRM2, DSCAL
*  =============================================================
*
*     .. Parameters ..
      real   ONE, ZERO
      PARAMETER        ( ONE = 1.0D+0 , ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, P, PHAT, Q, QHAT, U, UHAT, VHAT,
     $                   MAXIT, NEED1, NEED2
      real   TOL,  BNRM2, RHOTOL, 
     $       sGETBREAK, 
     $       sNRM2

      real   ALPHA, BETA, RHO, RHO1, TMPVAL,
     $     sdot
*     ..
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     .. External Funcs ..
      EXTERNAL           sGETBREAK, sAXPY, 
     $     sCOPY, sdot, sNRM2, sSCAL
*     ..
*     .. Intrinsic Funcs ..
      INTRINSIC          ABS, MAX
*     ..
*     .. Executable Statements ..
*
*     Entry point, test IJOB
      IF (IJOB .eq. 1) THEN
         GOTO 1
      ELSEIF (IJOB .eq. 2) THEN
*        here we do resumption handling
         IF (RLBL .eq. 2) GOTO 2
         IF (RLBL .eq. 3) GOTO 3
         IF (RLBL .eq. 4) GOTO 4
         IF (RLBL .eq. 5) GOTO 5
         IF (RLBL .eq. 6) GOTO 6
         IF (RLBL .eq. 7) GOTO 7
*        if neither of these, then error
         INFO = -6
         GOTO 20
      ENDIF
*
*
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      TOL   = RESID
*
*     Alias workspace columns.
*
      R    = 1
      RTLD = 2
      P    = 3
      PHAT = 4
      Q    = 5
      QHAT = 6
      U    = 6
      UHAT = 7
      VHAT = 7
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((U - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.9 ) THEN
            NEED1 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((U - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.9 ) THEN
            NEED2 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown tolerance parameter.
*
      RHOTOL = sGETBREAK()
*
*     Set initial residual.
*
      CALL sCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( sNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using RTLD[] as temp. storage.
*********CALL sCOPY(N, X, 1, WORK(1,RTLD), 1)
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 3
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      IF ( sNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      BNRM2 = sNRM2( N, B, 1 )
      IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
*
*     Choose RTLD such that initially, (R,RTLD) = RHO is not equal to 0.
*     Here we choose RTLD = R.
*
      CALL sCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform Conjugate Gradient Squared iteration.
*
         ITER = ITER + 1
*
         RHO = sdot( N, WORK(1,RTLD), 1, WORK(1,R), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
*        Compute direction vectors U and P.
*
         IF ( ITER.GT.1 ) THEN
*
*           Compute U.
*
            BETA = RHO / RHO1
            CALL sCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL sAXPY( N, BETA, WORK(1,Q), 1, WORK(1,U), 1 )
*
*           Compute P.
*
            CALL sSCAL( N, BETA**2, WORK(1,P), 1 )
            CALL sAXPY( N, BETA, WORK(1,Q), 1, WORK(1,P), 1 )
            TMPVAL = ONE
            CALL sAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,P), 1 )
         ELSE
            CALL sCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL sCOPY( N, WORK(1,U), 1, WORK(1,P), 1 )
         ENDIF
*
*        Compute direction adjusting scalar ALPHA.
*
*********CALL PSOLVE( WORK(1,PHAT), WORK(1,P) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((P    - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
*********CALL MATVEC( ONE, WORK(1,PHAT), ZERO, WORK(1,VHAT) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((VHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 1
         RETURN
*
*****************
 4       CONTINUE
*****************
*
         TMPVAL = sdot( N, WORK(1,RTLD), 1, WORK(1,VHAT), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
         CALL sCOPY( N, WORK(1,U), 1, WORK(1,Q), 1 )
         CALL sAXPY( N, -ALPHA, WORK(1,VHAT), 1, WORK(1,Q), 1 )
*
*        Compute direction adjusting vectORT UHAT.
*        PHAT is being used as temporary storage here.
*
         CALL sCOPY( N, WORK(1,Q), 1, WORK(1,PHAT), 1 ) 
         TMPVAL = ONE
         CALL sAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,PHAT), 1 )
*********CALL PSOLVE( WORK(1,UHAT), WORK(1,PHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((PHAT - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*        Compute new solution approximation vector X.
*
         CALL sAXPY( N, ALPHA, WORK(1,UHAT), 1, X, 1 )
*
*        Compute residual R and check for tolerance.
*
*********CALL MATVEC( ONE, WORK(1,UHAT), ZERO, WORK(1,QHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((QHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         CALL sAXPY( N, -ALPHA, WORK(1,QHAT), 1, WORK(1,R), 1 )
*
*********RESID = sNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 4
         RETURN
*
*****************
 7       CONTINUE
*****************
         IF( INFO.EQ.1 ) GO TO 30
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 20
         ENDIF
*
         RHO1 = RHO
*
      GO TO 10
*
   20 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
   25 CONTINUE
*
*     Set breakdown flag.
*
      IF ( ABS( RHO ).LT.RHOTOL ) THEN
         INFO = -10
         RLBL = -1
         IJOB = -1
         RETURN
      ENDIF
*
   30 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1
      RETURN
*
*     End of CGSREVCOM
*
      END
*     END SUBROUTINE sCGSREVCOM







      SUBROUTINE dCGSREVCOM(N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                     NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the 
*     Solution of Linear Systems: Building Blocks for Iterative 
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra, 
*     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, LDW, ITER, INFO
      double precision    RESID
      INTEGER            NDX1, NDX2
      double precision   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      double precision   X( * ), B( * ), WORK( LDW,* )
*     ..
*
*  Purpose
*  =======
*
*  CGS solves the linear system Ax = b using the
*  Conjugate Gradient Squared iterative method with preconditioning.
*
*  Convergence test: ( norm( b - A*x ) / norm( b ) ) .ls.  TOL.
*  For other measures, see the above reference.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER. 
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
* 
*  B       (input) DOUBLE PRECISION array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess. This is commonly set to
*          the zero vector. The user should be warned that for
*          this particular algorithm, an initial guess close to
*          the actual solution can result in divergence.
*          On exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,7)
*          Workspace for residual, direction vector, etc.
*          Note that vectors PHAT and QHAT, and UHAT and VHAT share 
*          the same workspace.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input) DOUBLE PRECISION
*          On input, the allowable convergence measure for
*          norm( b - A*x ).
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit.
*          .gt.   0: Convergence not achieved. This will be set
*                to the number of iterations performed.
*
*          .ls.   0: Illegal input parameter, or breakdown occurred
*                during iteration.
*
*                Illegal parameter: 
*
*                   -1: matrix dimension N .ls.  0
*                   -2: LDW .ls.  N
*                   -3: Maximum number of iterations ITER .ls. = 0.
*                   -5: Erroneous NDX1/NDX2 in INIT call.
*                   -6: Erroneous RLBL.
*
*                BREAKDOWN: If RHO become smaller than some tolerance,
*                   the program will terminate. Here we check 
*                   against tolerance BREAKTOL.
*
*                   -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                        orthogonal.
*
*  NDX1    (input/output) INTEGER. 
*  NDX2    On entry in INIT call contain indices required by interface
*          level for stopping test.
*          All other times, used as output, to indicate indices into
*          WORK[] for the MATVEC, PSOLVE done by the interface level.
*
*  SCLR1   (output) DOUBLE PRECISION.
*  SCLR2   Used to pass the scalars used in MATVEC. Scalars are reqd because
*          original routines use dgemv.
*
*  IJOB    (input/output) INTEGER. 
*          Used to communicate job code between the two levels.
*
*  BLAS CALLS:    DAXPY, DCOPY, DDOT, DNRM2, DSCAL
*  =============================================================
*
*     .. Parameters ..
      double precision   ONE, ZERO
      PARAMETER        ( ONE = 1.0D+0 , ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, P, PHAT, Q, QHAT, U, UHAT, VHAT,
     $                   MAXIT, NEED1, NEED2
      double precision   TOL,  BNRM2, RHOTOL, 
     $       dGETBREAK, 
     $       dNRM2

      double precision   ALPHA, BETA, RHO, RHO1, TMPVAL,
     $     ddot
*     ..
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     .. External Funcs ..
      EXTERNAL           dGETBREAK, dAXPY, 
     $     dCOPY, ddot, dNRM2, dSCAL
*     ..
*     .. Intrinsic Funcs ..
      INTRINSIC          ABS, MAX
*     ..
*     .. Executable Statements ..
*
*     Entry point, test IJOB
      IF (IJOB .eq. 1) THEN
         GOTO 1
      ELSEIF (IJOB .eq. 2) THEN
*        here we do resumption handling
         IF (RLBL .eq. 2) GOTO 2
         IF (RLBL .eq. 3) GOTO 3
         IF (RLBL .eq. 4) GOTO 4
         IF (RLBL .eq. 5) GOTO 5
         IF (RLBL .eq. 6) GOTO 6
         IF (RLBL .eq. 7) GOTO 7
*        if neither of these, then error
         INFO = -6
         GOTO 20
      ENDIF
*
*
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      TOL   = RESID
*
*     Alias workspace columns.
*
      R    = 1
      RTLD = 2
      P    = 3
      PHAT = 4
      Q    = 5
      QHAT = 6
      U    = 6
      UHAT = 7
      VHAT = 7
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((U - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.9 ) THEN
            NEED1 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((U - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.9 ) THEN
            NEED2 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown tolerance parameter.
*
      RHOTOL = dGETBREAK()
*
*     Set initial residual.
*
      CALL dCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( dNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using RTLD[] as temp. storage.
*********CALL dCOPY(N, X, 1, WORK(1,RTLD), 1)
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 3
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      IF ( dNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      BNRM2 = dNRM2( N, B, 1 )
      IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
*
*     Choose RTLD such that initially, (R,RTLD) = RHO is not equal to 0.
*     Here we choose RTLD = R.
*
      CALL dCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform Conjugate Gradient Squared iteration.
*
         ITER = ITER + 1
*
         RHO = ddot( N, WORK(1,RTLD), 1, WORK(1,R), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
*        Compute direction vectors U and P.
*
         IF ( ITER.GT.1 ) THEN
*
*           Compute U.
*
            BETA = RHO / RHO1
            CALL dCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL dAXPY( N, BETA, WORK(1,Q), 1, WORK(1,U), 1 )
*
*           Compute P.
*
            CALL dSCAL( N, BETA**2, WORK(1,P), 1 )
            CALL dAXPY( N, BETA, WORK(1,Q), 1, WORK(1,P), 1 )
            TMPVAL = ONE
            CALL dAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,P), 1 )
         ELSE
            CALL dCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL dCOPY( N, WORK(1,U), 1, WORK(1,P), 1 )
         ENDIF
*
*        Compute direction adjusting scalar ALPHA.
*
*********CALL PSOLVE( WORK(1,PHAT), WORK(1,P) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((P    - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
*********CALL MATVEC( ONE, WORK(1,PHAT), ZERO, WORK(1,VHAT) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((VHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 1
         RETURN
*
*****************
 4       CONTINUE
*****************
*
         TMPVAL = ddot( N, WORK(1,RTLD), 1, WORK(1,VHAT), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
         CALL dCOPY( N, WORK(1,U), 1, WORK(1,Q), 1 )
         CALL dAXPY( N, -ALPHA, WORK(1,VHAT), 1, WORK(1,Q), 1 )
*
*        Compute direction adjusting vectORT UHAT.
*        PHAT is being used as temporary storage here.
*
         CALL dCOPY( N, WORK(1,Q), 1, WORK(1,PHAT), 1 ) 
         TMPVAL = ONE
         CALL dAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,PHAT), 1 )
*********CALL PSOLVE( WORK(1,UHAT), WORK(1,PHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((PHAT - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*        Compute new solution approximation vector X.
*
         CALL dAXPY( N, ALPHA, WORK(1,UHAT), 1, X, 1 )
*
*        Compute residual R and check for tolerance.
*
*********CALL MATVEC( ONE, WORK(1,UHAT), ZERO, WORK(1,QHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((QHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         CALL dAXPY( N, -ALPHA, WORK(1,QHAT), 1, WORK(1,R), 1 )
*
*********RESID = dNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 4
         RETURN
*
*****************
 7       CONTINUE
*****************
         IF( INFO.EQ.1 ) GO TO 30
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 20
         ENDIF
*
         RHO1 = RHO
*
      GO TO 10
*
   20 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
   25 CONTINUE
*
*     Set breakdown flag.
*
      IF ( ABS( RHO ).LT.RHOTOL ) THEN
         INFO = -10
         RLBL = -1
         IJOB = -1
         RETURN
      ENDIF
*
   30 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1
      RETURN
*
*     End of CGSREVCOM
*
      END
*     END SUBROUTINE dCGSREVCOM







      SUBROUTINE cCGSREVCOM(N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                     NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the 
*     Solution of Linear Systems: Building Blocks for Iterative 
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra, 
*     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, LDW, ITER, INFO
      real    RESID
      INTEGER            NDX1, NDX2
      complex   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      complex   X( * ), B( * ), WORK( LDW,* )
*     ..
*
*  Purpose
*  =======
*
*  CGS solves the linear system Ax = b using the
*  Conjugate Gradient Squared iterative method with preconditioning.
*
*  Convergence test: ( norm( b - A*x ) / norm( b ) ) .ls.  TOL.
*  For other measures, see the above reference.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER. 
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
* 
*  B       (input) DOUBLE PRECISION array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess. This is commonly set to
*          the zero vector. The user should be warned that for
*          this particular algorithm, an initial guess close to
*          the actual solution can result in divergence.
*          On exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,7)
*          Workspace for residual, direction vector, etc.
*          Note that vectors PHAT and QHAT, and UHAT and VHAT share 
*          the same workspace.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input) DOUBLE PRECISION
*          On input, the allowable convergence measure for
*          norm( b - A*x ).
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit.
*          .gt.   0: Convergence not achieved. This will be set
*                to the number of iterations performed.
*
*          .ls.   0: Illegal input parameter, or breakdown occurred
*                during iteration.
*
*                Illegal parameter: 
*
*                   -1: matrix dimension N .ls.  0
*                   -2: LDW .ls.  N
*                   -3: Maximum number of iterations ITER .ls. = 0.
*                   -5: Erroneous NDX1/NDX2 in INIT call.
*                   -6: Erroneous RLBL.
*
*                BREAKDOWN: If RHO become smaller than some tolerance,
*                   the program will terminate. Here we check 
*                   against tolerance BREAKTOL.
*
*                   -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                        orthogonal.
*
*  NDX1    (input/output) INTEGER. 
*  NDX2    On entry in INIT call contain indices required by interface
*          level for stopping test.
*          All other times, used as output, to indicate indices into
*          WORK[] for the MATVEC, PSOLVE done by the interface level.
*
*  SCLR1   (output) DOUBLE PRECISION.
*  SCLR2   Used to pass the scalars used in MATVEC. Scalars are reqd because
*          original routines use dgemv.
*
*  IJOB    (input/output) INTEGER. 
*          Used to communicate job code between the two levels.
*
*  BLAS CALLS:    DAXPY, DCOPY, DDOT, DNRM2, DSCAL
*  =============================================================
*
*     .. Parameters ..
      real   ONE, ZERO
      PARAMETER        ( ONE = 1.0D+0 , ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, P, PHAT, Q, QHAT, U, UHAT, VHAT,
     $                   MAXIT, NEED1, NEED2
      real   TOL,  BNRM2, RHOTOL, 
     $       sGETBREAK, 
     $       scNRM2

      complex   ALPHA, BETA, RHO, RHO1, TMPVAL,
     $     wcdotc
*     ..
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     .. External Funcs ..
      EXTERNAL           sGETBREAK, cAXPY, 
     $     cCOPY, wcdotc, scNRM2, cSCAL
*     ..
*     .. Intrinsic Funcs ..
      INTRINSIC          ABS, MAX
*     ..
*     .. Executable Statements ..
*
*     Entry point, test IJOB
      IF (IJOB .eq. 1) THEN
         GOTO 1
      ELSEIF (IJOB .eq. 2) THEN
*        here we do resumption handling
         IF (RLBL .eq. 2) GOTO 2
         IF (RLBL .eq. 3) GOTO 3
         IF (RLBL .eq. 4) GOTO 4
         IF (RLBL .eq. 5) GOTO 5
         IF (RLBL .eq. 6) GOTO 6
         IF (RLBL .eq. 7) GOTO 7
*        if neither of these, then error
         INFO = -6
         GOTO 20
      ENDIF
*
*
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      TOL   = RESID
*
*     Alias workspace columns.
*
      R    = 1
      RTLD = 2
      P    = 3
      PHAT = 4
      Q    = 5
      QHAT = 6
      U    = 6
      UHAT = 7
      VHAT = 7
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((U - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.9 ) THEN
            NEED1 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((U - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.9 ) THEN
            NEED2 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown tolerance parameter.
*
      RHOTOL = sGETBREAK()
*
*     Set initial residual.
*
      CALL cCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( scNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using RTLD[] as temp. storage.
*********CALL cCOPY(N, X, 1, WORK(1,RTLD), 1)
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 3
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      IF ( scNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      BNRM2 = scNRM2( N, B, 1 )
      IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
*
*     Choose RTLD such that initially, (R,RTLD) = RHO is not equal to 0.
*     Here we choose RTLD = R.
*
      CALL cCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform Conjugate Gradient Squared iteration.
*
         ITER = ITER + 1
*
         RHO = wcdotc( N, WORK(1,RTLD), 1, WORK(1,R), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
*        Compute direction vectors U and P.
*
         IF ( ITER.GT.1 ) THEN
*
*           Compute U.
*
            BETA = RHO / RHO1
            CALL cCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL cAXPY( N, BETA, WORK(1,Q), 1, WORK(1,U), 1 )
*
*           Compute P.
*
            CALL cSCAL( N, BETA**2, WORK(1,P), 1 )
            CALL cAXPY( N, BETA, WORK(1,Q), 1, WORK(1,P), 1 )
            TMPVAL = ONE
            CALL cAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,P), 1 )
         ELSE
            CALL cCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL cCOPY( N, WORK(1,U), 1, WORK(1,P), 1 )
         ENDIF
*
*        Compute direction adjusting scalar ALPHA.
*
*********CALL PSOLVE( WORK(1,PHAT), WORK(1,P) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((P    - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
*********CALL MATVEC( ONE, WORK(1,PHAT), ZERO, WORK(1,VHAT) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((VHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 1
         RETURN
*
*****************
 4       CONTINUE
*****************
*
         TMPVAL = wcdotc( N, WORK(1,RTLD), 1, WORK(1,VHAT), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
         CALL cCOPY( N, WORK(1,U), 1, WORK(1,Q), 1 )
         CALL cAXPY( N, -ALPHA, WORK(1,VHAT), 1, WORK(1,Q), 1 )
*
*        Compute direction adjusting vectORT UHAT.
*        PHAT is being used as temporary storage here.
*
         CALL cCOPY( N, WORK(1,Q), 1, WORK(1,PHAT), 1 ) 
         TMPVAL = ONE
         CALL cAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,PHAT), 1 )
*********CALL PSOLVE( WORK(1,UHAT), WORK(1,PHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((PHAT - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*        Compute new solution approximation vector X.
*
         CALL cAXPY( N, ALPHA, WORK(1,UHAT), 1, X, 1 )
*
*        Compute residual R and check for tolerance.
*
*********CALL MATVEC( ONE, WORK(1,UHAT), ZERO, WORK(1,QHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((QHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         CALL cAXPY( N, -ALPHA, WORK(1,QHAT), 1, WORK(1,R), 1 )
*
*********RESID = scNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 4
         RETURN
*
*****************
 7       CONTINUE
*****************
         IF( INFO.EQ.1 ) GO TO 30
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 20
         ENDIF
*
         RHO1 = RHO
*
      GO TO 10
*
   20 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
   25 CONTINUE
*
*     Set breakdown flag.
*
      IF ( ABS( RHO ).LT.RHOTOL ) THEN
         INFO = -10
         RLBL = -1
         IJOB = -1
         RETURN
      ENDIF
*
   30 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1
      RETURN
*
*     End of CGSREVCOM
*
      END
*     END SUBROUTINE cCGSREVCOM







      SUBROUTINE zCGSREVCOM(N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                     NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the 
*     Solution of Linear Systems: Building Blocks for Iterative 
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra, 
*     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, LDW, ITER, INFO
      double precision    RESID
      INTEGER            NDX1, NDX2
      double complex   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      double complex   X( * ), B( * ), WORK( LDW,* )
*     ..
*
*  Purpose
*  =======
*
*  CGS solves the linear system Ax = b using the
*  Conjugate Gradient Squared iterative method with preconditioning.
*
*  Convergence test: ( norm( b - A*x ) / norm( b ) ) .ls.  TOL.
*  For other measures, see the above reference.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER. 
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
* 
*  B       (input) DOUBLE PRECISION array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess. This is commonly set to
*          the zero vector. The user should be warned that for
*          this particular algorithm, an initial guess close to
*          the actual solution can result in divergence.
*          On exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,7)
*          Workspace for residual, direction vector, etc.
*          Note that vectors PHAT and QHAT, and UHAT and VHAT share 
*          the same workspace.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input) DOUBLE PRECISION
*          On input, the allowable convergence measure for
*          norm( b - A*x ).
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit.
*          .gt.   0: Convergence not achieved. This will be set
*                to the number of iterations performed.
*
*          .ls.   0: Illegal input parameter, or breakdown occurred
*                during iteration.
*
*                Illegal parameter: 
*
*                   -1: matrix dimension N .ls.  0
*                   -2: LDW .ls.  N
*                   -3: Maximum number of iterations ITER .ls. = 0.
*                   -5: Erroneous NDX1/NDX2 in INIT call.
*                   -6: Erroneous RLBL.
*
*                BREAKDOWN: If RHO become smaller than some tolerance,
*                   the program will terminate. Here we check 
*                   against tolerance BREAKTOL.
*
*                   -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                        orthogonal.
*
*  NDX1    (input/output) INTEGER. 
*  NDX2    On entry in INIT call contain indices required by interface
*          level for stopping test.
*          All other times, used as output, to indicate indices into
*          WORK[] for the MATVEC, PSOLVE done by the interface level.
*
*  SCLR1   (output) DOUBLE PRECISION.
*  SCLR2   Used to pass the scalars used in MATVEC. Scalars are reqd because
*          original routines use dgemv.
*
*  IJOB    (input/output) INTEGER. 
*          Used to communicate job code between the two levels.
*
*  BLAS CALLS:    DAXPY, DCOPY, DDOT, DNRM2, DSCAL
*  =============================================================
*
*     .. Parameters ..
      double precision   ONE, ZERO
      PARAMETER        ( ONE = 1.0D+0 , ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, P, PHAT, Q, QHAT, U, UHAT, VHAT,
     $                   MAXIT, NEED1, NEED2
      double precision   TOL,  BNRM2, RHOTOL, 
     $       dGETBREAK, 
     $       dzNRM2

      double complex   ALPHA, BETA, RHO, RHO1, TMPVAL,
     $     wzdotc
*     ..
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     .. External Funcs ..
      EXTERNAL           dGETBREAK, zAXPY, 
     $     zCOPY, wzdotc, dzNRM2, zSCAL
*     ..
*     .. Intrinsic Funcs ..
      INTRINSIC          ABS, MAX
*     ..
*     .. Executable Statements ..
*
*     Entry point, test IJOB
      IF (IJOB .eq. 1) THEN
         GOTO 1
      ELSEIF (IJOB .eq. 2) THEN
*        here we do resumption handling
         IF (RLBL .eq. 2) GOTO 2
         IF (RLBL .eq. 3) GOTO 3
         IF (RLBL .eq. 4) GOTO 4
         IF (RLBL .eq. 5) GOTO 5
         IF (RLBL .eq. 6) GOTO 6
         IF (RLBL .eq. 7) GOTO 7
*        if neither of these, then error
         INFO = -6
         GOTO 20
      ENDIF
*
*
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      TOL   = RESID
*
*     Alias workspace columns.
*
      R    = 1
      RTLD = 2
      P    = 3
      PHAT = 4
      Q    = 5
      QHAT = 6
      U    = 6
      UHAT = 7
      VHAT = 7
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((U - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.9 ) THEN
            NEED1 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((PHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((QHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((U - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((UHAT - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.9 ) THEN
            NEED2 = ((VHAT - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown tolerance parameter.
*
      RHOTOL = dGETBREAK()
*
*     Set initial residual.
*
      CALL zCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( dzNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using RTLD[] as temp. storage.
*********CALL zCOPY(N, X, 1, WORK(1,RTLD), 1)
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 3
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      IF ( dzNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      BNRM2 = dzNRM2( N, B, 1 )
      IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
*
*     Choose RTLD such that initially, (R,RTLD) = RHO is not equal to 0.
*     Here we choose RTLD = R.
*
      CALL zCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform Conjugate Gradient Squared iteration.
*
         ITER = ITER + 1
*
         RHO = wzdotc( N, WORK(1,RTLD), 1, WORK(1,R), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
*        Compute direction vectors U and P.
*
         IF ( ITER.GT.1 ) THEN
*
*           Compute U.
*
            BETA = RHO / RHO1
            CALL zCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL zAXPY( N, BETA, WORK(1,Q), 1, WORK(1,U), 1 )
*
*           Compute P.
*
            CALL zSCAL( N, BETA**2, WORK(1,P), 1 )
            CALL zAXPY( N, BETA, WORK(1,Q), 1, WORK(1,P), 1 )
            TMPVAL = ONE
            CALL zAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,P), 1 )
         ELSE
            CALL zCOPY( N, WORK(1,R), 1, WORK(1,U), 1 )
            CALL zCOPY( N, WORK(1,U), 1, WORK(1,P), 1 )
         ENDIF
*
*        Compute direction adjusting scalar ALPHA.
*
*********CALL PSOLVE( WORK(1,PHAT), WORK(1,P) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((P    - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
*********CALL MATVEC( ONE, WORK(1,PHAT), ZERO, WORK(1,VHAT) )
*
         NDX1 = ((PHAT - 1) * LDW) + 1
         NDX2 = ((VHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 1
         RETURN
*
*****************
 4       CONTINUE
*****************
*
         TMPVAL = wzdotc( N, WORK(1,RTLD), 1, WORK(1,VHAT), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
         CALL zCOPY( N, WORK(1,U), 1, WORK(1,Q), 1 )
         CALL zAXPY( N, -ALPHA, WORK(1,VHAT), 1, WORK(1,Q), 1 )
*
*        Compute direction adjusting vectORT UHAT.
*        PHAT is being used as temporary storage here.
*
         CALL zCOPY( N, WORK(1,Q), 1, WORK(1,PHAT), 1 ) 
         TMPVAL = ONE
         CALL zAXPY( N, TMPVAL, WORK(1,U), 1, WORK(1,PHAT), 1 )
*********CALL PSOLVE( WORK(1,UHAT), WORK(1,PHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((PHAT - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*        Compute new solution approximation vector X.
*
         CALL zAXPY( N, ALPHA, WORK(1,UHAT), 1, X, 1 )
*
*        Compute residual R and check for tolerance.
*
*********CALL MATVEC( ONE, WORK(1,UHAT), ZERO, WORK(1,QHAT) )
*
         NDX1 = ((UHAT - 1) * LDW) + 1
         NDX2 = ((QHAT - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         CALL zAXPY( N, -ALPHA, WORK(1,QHAT), 1, WORK(1,R), 1 )
*
*********RESID = dzNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 4
         RETURN
*
*****************
 7       CONTINUE
*****************
         IF( INFO.EQ.1 ) GO TO 30
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 20
         ENDIF
*
         RHO1 = RHO
*
      GO TO 10
*
   20 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
   25 CONTINUE
*
*     Set breakdown flag.
*
      IF ( ABS( RHO ).LT.RHOTOL ) THEN
         INFO = -10
         RLBL = -1
         IJOB = -1
         RETURN
      ENDIF
*
   30 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1
      RETURN
*
*     End of CGSREVCOM
*
      END
*     END SUBROUTINE zCGSREVCOM







