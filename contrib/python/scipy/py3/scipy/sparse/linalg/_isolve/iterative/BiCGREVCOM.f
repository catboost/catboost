*  -*- fortran -*-
      SUBROUTINE sBICGREVCOM( N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                       NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
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
      real  RESID
      INTEGER            NDX1, NDX2
      real   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      real   X( * ), B( * ), WORK( LDW,* )
*
*     ..
*
*  Purpose
*  =======
*
*  BiCG solves the linear system Ax = b using the
*  BiConjugate Gradient iterative method with preconditioning.
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
*  X      (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6).
*          Workspace for residual, direction vector, etc.
*          Note that Z and Q, and ZTLD and QTLD share workspace.
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
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          .gt.   0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
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
*                BREAKDOWN: If parameters RHO or OMEGA become smaller
*                   than some tolerance, the program will terminate.
*                   Here we check against tolerance BREAKTOL.
*
*                  -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                       orthogonal.
*
*                  BREAKTOL is set in func GETBREAK.
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
*  BLAS CALLS:   DAXPY, DCOPY, DDOT, DNRM2,
*  ==============================================================
*
*     .. Parameters ..
      real   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, Z, ZTLD, P, PTLD, Q, QTLD, MAXIT,
     $                   NEED1, NEED2
      real   TOL, RHOTOL, 
     $       sGETBREAK,  
     $       sNRM2
      real  ALPHA, BETA, RHO, RHO1, sdot,
     $     TMPVAL
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL           sAXPY, sCOPY, sdot, sNRM2
*     ..
*     .. Executable Statements ..
*
*     Entry point, so test IJOB
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
* init.
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
      Z    = 3
      ZTLD = 4
      P    = 5
      PTLD = 6
      Q    = 3
      QTLD = 4
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((QTLD - 1) * LDW) + 1
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
            NEED2 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((QTLD - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown parameters.
*
      RHOTOL = sGETBREAK()
* 
*     Set initial residual.
*
      CALL sCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( sNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        using WORK[RTLD] as temp
*********CALL sCOPY( N, X, 1, WORK(1,RTLD), 1 )
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
         RLBL = 2
         IJOB = 5
         RETURN
      ENDIF
*****************
 2    CONTINUE
*****************
*
      IF ( sNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      CALL sCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform BiConjugate Gradient iteration.
*
         ITER = ITER + 1
*
*        Compute direction vectors PK and PTLD.
*
*********CALL PSOLVE( WORK(1,Z), WORK(1,R) )
*
         NDX1 = ((Z - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
         RLBL = 3
         IJOB = 3
         RETURN
*****************
 3       CONTINUE
*****************
*
*********CALL PSOLVETRANS( WORK(1,ZTLD), WORK(1,RTLD) )
*
         NDX1 = ((ZTLD - 1) * LDW) + 1
         NDX2 = ((RTLD - 1) * LDW) + 1
         RLBL = 4
         IJOB = 4
         RETURN
*****************
 4       CONTINUE
*****************
*
*
*         RHO = sdot( N, WORK(1,Z), 1, WORK(1,RTLD), 1 )
         RHO = sdot( N, WORK(1,RTLD), 1, WORK(1,Z), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
         IF ( ITER.GT.1 ) THEN
            BETA = RHO / RHO1
            CALL sAXPY( N, BETA, WORK(1,P), 1, WORK(1,Z), 1 )
*            CALL sAXPY( N, BETA, WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL sAXPY( N, (BETA), 
     $           WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL sCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL sCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ELSE
            CALL sCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL sCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ENDIF
*
*********CALL MATVEC( ONE, WORK(1,P), ZERO, WORK(1,Q) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((P - 1) * LDW) + 1
         NDX2 = ((Q - 1) * LDW) + 1
         RLBL = 5
         IJOB = 1
         RETURN
*****************
 5       CONTINUE
*****************
*
*********CALL MATVECTRANS( ONE, WORK(1,PTLD), ZERO, WORK(1,QTLD) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((PTLD - 1) * LDW) + 1
         NDX2 = ((QTLD - 1) * LDW) + 1
         RLBL = 6
         IJOB = 2
         RETURN
*****************
 6       CONTINUE
*****************
         TMPVAL = sdot( N, WORK(1,PTLD), 1, WORK(1,Q), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
*        Compute current solution vector x.
*
         CALL sAXPY( N, ALPHA, WORK(1,P), 1, X, 1 )
*
*        Compute residual vector rk, find norm,
*        then check for tolerance.
*
         CALL sAXPY( N, -ALPHA, WORK(1,Q), 1, WORK(1,R), 1 )
*
*********RESID = sNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 6
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
*         CALL sAXPY( N, -ALPHA, WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
         CALL sAXPY( N, -(ALPHA)
     $        , WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
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
      INFO = -10
      RLBL = -1
      IJOB = -1
      RETURN
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
*     End of BICGREVCOM
*
      END
*     END SUBROUTINE sBICGREVCOM







      SUBROUTINE dBICGREVCOM( N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                       NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
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
      double precision  RESID
      INTEGER            NDX1, NDX2
      double precision   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      double precision   X( * ), B( * ), WORK( LDW,* )
*
*     ..
*
*  Purpose
*  =======
*
*  BiCG solves the linear system Ax = b using the
*  BiConjugate Gradient iterative method with preconditioning.
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
*  X      (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6).
*          Workspace for residual, direction vector, etc.
*          Note that Z and Q, and ZTLD and QTLD share workspace.
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
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          .gt.   0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
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
*                BREAKDOWN: If parameters RHO or OMEGA become smaller
*                   than some tolerance, the program will terminate.
*                   Here we check against tolerance BREAKTOL.
*
*                  -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                       orthogonal.
*
*                  BREAKTOL is set in func GETBREAK.
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
*  BLAS CALLS:   DAXPY, DCOPY, DDOT, DNRM2,
*  ==============================================================
*
*     .. Parameters ..
      double precision   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, Z, ZTLD, P, PTLD, Q, QTLD, MAXIT,
     $                   NEED1, NEED2
      double precision   TOL, RHOTOL, 
     $       dGETBREAK,  
     $       dNRM2
      double precision  ALPHA, BETA, RHO, RHO1, ddot,
     $     TMPVAL
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL           dAXPY, dCOPY, ddot, dNRM2
*     ..
*     .. Executable Statements ..
*
*     Entry point, so test IJOB
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
* init.
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
      Z    = 3
      ZTLD = 4
      P    = 5
      PTLD = 6
      Q    = 3
      QTLD = 4
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((QTLD - 1) * LDW) + 1
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
            NEED2 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((QTLD - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown parameters.
*
      RHOTOL = dGETBREAK()
* 
*     Set initial residual.
*
      CALL dCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( dNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        using WORK[RTLD] as temp
*********CALL dCOPY( N, X, 1, WORK(1,RTLD), 1 )
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
         RLBL = 2
         IJOB = 5
         RETURN
      ENDIF
*****************
 2    CONTINUE
*****************
*
      IF ( dNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      CALL dCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform BiConjugate Gradient iteration.
*
         ITER = ITER + 1
*
*        Compute direction vectors PK and PTLD.
*
*********CALL PSOLVE( WORK(1,Z), WORK(1,R) )
*
         NDX1 = ((Z - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
         RLBL = 3
         IJOB = 3
         RETURN
*****************
 3       CONTINUE
*****************
*
*********CALL PSOLVETRANS( WORK(1,ZTLD), WORK(1,RTLD) )
*
         NDX1 = ((ZTLD - 1) * LDW) + 1
         NDX2 = ((RTLD - 1) * LDW) + 1
         RLBL = 4
         IJOB = 4
         RETURN
*****************
 4       CONTINUE
*****************
*
*
*         RHO = ddot( N, WORK(1,Z), 1, WORK(1,RTLD), 1 )
         RHO = ddot( N, WORK(1,RTLD), 1, WORK(1,Z), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
         IF ( ITER.GT.1 ) THEN
            BETA = RHO / RHO1
            CALL dAXPY( N, BETA, WORK(1,P), 1, WORK(1,Z), 1 )
*            CALL dAXPY( N, BETA, WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL dAXPY( N, (BETA), 
     $           WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL dCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL dCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ELSE
            CALL dCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL dCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ENDIF
*
*********CALL MATVEC( ONE, WORK(1,P), ZERO, WORK(1,Q) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((P - 1) * LDW) + 1
         NDX2 = ((Q - 1) * LDW) + 1
         RLBL = 5
         IJOB = 1
         RETURN
*****************
 5       CONTINUE
*****************
*
*********CALL MATVECTRANS( ONE, WORK(1,PTLD), ZERO, WORK(1,QTLD) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((PTLD - 1) * LDW) + 1
         NDX2 = ((QTLD - 1) * LDW) + 1
         RLBL = 6
         IJOB = 2
         RETURN
*****************
 6       CONTINUE
*****************
         TMPVAL = ddot( N, WORK(1,PTLD), 1, WORK(1,Q), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
*        Compute current solution vector x.
*
         CALL dAXPY( N, ALPHA, WORK(1,P), 1, X, 1 )
*
*        Compute residual vector rk, find norm,
*        then check for tolerance.
*
         CALL dAXPY( N, -ALPHA, WORK(1,Q), 1, WORK(1,R), 1 )
*
*********RESID = dNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 6
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
*         CALL dAXPY( N, -ALPHA, WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
         CALL dAXPY( N, -(ALPHA)
     $        , WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
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
      INFO = -10
      RLBL = -1
      IJOB = -1
      RETURN
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
*     End of BICGREVCOM
*
      END
*     END SUBROUTINE dBICGREVCOM







      SUBROUTINE cBICGREVCOM( N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                       NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
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
      real  RESID
      INTEGER            NDX1, NDX2
      complex   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      complex   X( * ), B( * ), WORK( LDW,* )
*
*     ..
*
*  Purpose
*  =======
*
*  BiCG solves the linear system Ax = b using the
*  BiConjugate Gradient iterative method with preconditioning.
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
*  X      (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6).
*          Workspace for residual, direction vector, etc.
*          Note that Z and Q, and ZTLD and QTLD share workspace.
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
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          .gt.   0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
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
*                BREAKDOWN: If parameters RHO or OMEGA become smaller
*                   than some tolerance, the program will terminate.
*                   Here we check against tolerance BREAKTOL.
*
*                  -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                       orthogonal.
*
*                  BREAKTOL is set in func GETBREAK.
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
*  BLAS CALLS:   DAXPY, DCOPY, DDOT, DNRM2,
*  ==============================================================
*
*     .. Parameters ..
      real   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, Z, ZTLD, P, PTLD, Q, QTLD, MAXIT,
     $                   NEED1, NEED2
      real   TOL, RHOTOL, 
     $       sGETBREAK,  
     $       scNRM2
      complex  ALPHA, BETA, RHO, RHO1, wcdotc,
     $     TMPVAL
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL           cAXPY, cCOPY, wcdotc, scNRM2
*     ..
*     .. Executable Statements ..
*
*     Entry point, so test IJOB
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
* init.
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
      Z    = 3
      ZTLD = 4
      P    = 5
      PTLD = 6
      Q    = 3
      QTLD = 4
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((QTLD - 1) * LDW) + 1
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
            NEED2 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((QTLD - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown parameters.
*
      RHOTOL = sGETBREAK()
* 
*     Set initial residual.
*
      CALL cCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( scNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        using WORK[RTLD] as temp
*********CALL cCOPY( N, X, 1, WORK(1,RTLD), 1 )
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
         RLBL = 2
         IJOB = 5
         RETURN
      ENDIF
*****************
 2    CONTINUE
*****************
*
      IF ( scNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      CALL cCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform BiConjugate Gradient iteration.
*
         ITER = ITER + 1
*
*        Compute direction vectors PK and PTLD.
*
*********CALL PSOLVE( WORK(1,Z), WORK(1,R) )
*
         NDX1 = ((Z - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
         RLBL = 3
         IJOB = 3
         RETURN
*****************
 3       CONTINUE
*****************
*
*********CALL PSOLVETRANS( WORK(1,ZTLD), WORK(1,RTLD) )
*
         NDX1 = ((ZTLD - 1) * LDW) + 1
         NDX2 = ((RTLD - 1) * LDW) + 1
         RLBL = 4
         IJOB = 4
         RETURN
*****************
 4       CONTINUE
*****************
*
*
*         RHO = wcdotc( N, WORK(1,Z), 1, WORK(1,RTLD), 1 )
         RHO = wcdotc( N, WORK(1,RTLD), 1, WORK(1,Z), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
         IF ( ITER.GT.1 ) THEN
            BETA = RHO / RHO1
            CALL cAXPY( N, BETA, WORK(1,P), 1, WORK(1,Z), 1 )
*            CALL cAXPY( N, BETA, WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL cAXPY( N, conjg(BETA), 
     $           WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL cCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL cCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ELSE
            CALL cCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL cCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ENDIF
*
*********CALL MATVEC( ONE, WORK(1,P), ZERO, WORK(1,Q) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((P - 1) * LDW) + 1
         NDX2 = ((Q - 1) * LDW) + 1
         RLBL = 5
         IJOB = 1
         RETURN
*****************
 5       CONTINUE
*****************
*
*********CALL MATVECTRANS( ONE, WORK(1,PTLD), ZERO, WORK(1,QTLD) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((PTLD - 1) * LDW) + 1
         NDX2 = ((QTLD - 1) * LDW) + 1
         RLBL = 6
         IJOB = 2
         RETURN
*****************
 6       CONTINUE
*****************
         TMPVAL = wcdotc( N, WORK(1,PTLD), 1, WORK(1,Q), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
*        Compute current solution vector x.
*
         CALL cAXPY( N, ALPHA, WORK(1,P), 1, X, 1 )
*
*        Compute residual vector rk, find norm,
*        then check for tolerance.
*
         CALL cAXPY( N, -ALPHA, WORK(1,Q), 1, WORK(1,R), 1 )
*
*********RESID = scNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 6
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
*         CALL cAXPY( N, -ALPHA, WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
         CALL cAXPY( N, -conjg(ALPHA)
     $        , WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
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
      INFO = -10
      RLBL = -1
      IJOB = -1
      RETURN
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
*     End of BICGREVCOM
*
      END
*     END SUBROUTINE cBICGREVCOM







      SUBROUTINE zBICGREVCOM( N, B, X, WORK, LDW, ITER, RESID, INFO,
     $                       NDX1, NDX2, SCLR1, SCLR2, IJOB)
*
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
      double precision  RESID
      INTEGER            NDX1, NDX2
      double complex   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      double complex   X( * ), B( * ), WORK( LDW,* )
*
*     ..
*
*  Purpose
*  =======
*
*  BiCG solves the linear system Ax = b using the
*  BiConjugate Gradient iterative method with preconditioning.
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
*  X      (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6).
*          Workspace for residual, direction vector, etc.
*          Note that Z and Q, and ZTLD and QTLD share workspace.
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
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          .gt.   0: Convergence to tolerance not achieved. This will be
*                set to the number of iterations performed.
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
*                BREAKDOWN: If parameters RHO or OMEGA become smaller
*                   than some tolerance, the program will terminate.
*                   Here we check against tolerance BREAKTOL.
*
*                  -10: RHO .ls.  BREAKTOL: RHO and RTLD have become
*                                       orthogonal.
*
*                  BREAKTOL is set in func GETBREAK.
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
*  BLAS CALLS:   DAXPY, DCOPY, DDOT, DNRM2,
*  ==============================================================
*
*     .. Parameters ..
      double precision   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            R, RTLD, Z, ZTLD, P, PTLD, Q, QTLD, MAXIT,
     $                   NEED1, NEED2
      double precision   TOL, RHOTOL, 
     $       dGETBREAK,  
     $       dzNRM2
      double complex  ALPHA, BETA, RHO, RHO1, wzdotc,
     $     TMPVAL
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL           zAXPY, zCOPY, wzdotc, dzNRM2
*     ..
*     .. Executable Statements ..
*
*     Entry point, so test IJOB
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
* init.
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
      Z    = 3
      ZTLD = 4
      P    = 5
      PTLD = 6
      Q    = 3
      QTLD = 4
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((RTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((P - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.7 ) THEN
            NEED1 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.8 ) THEN
            NEED1 = ((QTLD - 1) * LDW) + 1
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
            NEED2 = ((Z - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((ZTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((P - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((PTLD - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.7 ) THEN
            NEED2 = ((Q - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.8 ) THEN
            NEED2 = ((QTLD - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 20
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set breakdown parameters.
*
      RHOTOL = dGETBREAK()
* 
*     Set initial residual.
*
      CALL zCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( dzNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        using WORK[RTLD] as temp
*********CALL zCOPY( N, X, 1, WORK(1,RTLD), 1 )
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R    - 1) * LDW) + 1
         RLBL = 2
         IJOB = 5
         RETURN
      ENDIF
*****************
 2    CONTINUE
*****************
*
      IF ( dzNRM2( N, WORK(1,R), 1 ).LE.TOL ) GO TO 30
*
      CALL zCOPY( N, WORK(1,R), 1, WORK(1,RTLD), 1 )
*
      ITER = 0
*
   10 CONTINUE
*
*     Perform BiConjugate Gradient iteration.
*
         ITER = ITER + 1
*
*        Compute direction vectors PK and PTLD.
*
*********CALL PSOLVE( WORK(1,Z), WORK(1,R) )
*
         NDX1 = ((Z - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
         RLBL = 3
         IJOB = 3
         RETURN
*****************
 3       CONTINUE
*****************
*
*********CALL PSOLVETRANS( WORK(1,ZTLD), WORK(1,RTLD) )
*
         NDX1 = ((ZTLD - 1) * LDW) + 1
         NDX2 = ((RTLD - 1) * LDW) + 1
         RLBL = 4
         IJOB = 4
         RETURN
*****************
 4       CONTINUE
*****************
*
*
*         RHO = wzdotc( N, WORK(1,Z), 1, WORK(1,RTLD), 1 )
         RHO = wzdotc( N, WORK(1,RTLD), 1, WORK(1,Z), 1 )
         IF ( ABS( RHO ).LT.RHOTOL ) GO TO 25
*
         IF ( ITER.GT.1 ) THEN
            BETA = RHO / RHO1
            CALL zAXPY( N, BETA, WORK(1,P), 1, WORK(1,Z), 1 )
*            CALL zAXPY( N, BETA, WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL zAXPY( N, conjg(BETA), 
     $           WORK(1,PTLD), 1, WORK(1,ZTLD), 1 )
            CALL zCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL zCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ELSE
            CALL zCOPY( N, WORK(1,Z), 1, WORK(1,P), 1 )
            CALL zCOPY( N, WORK(1,ZTLD), 1, WORK(1,PTLD), 1 )
         ENDIF
*
*********CALL MATVEC( ONE, WORK(1,P), ZERO, WORK(1,Q) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((P - 1) * LDW) + 1
         NDX2 = ((Q - 1) * LDW) + 1
         RLBL = 5
         IJOB = 1
         RETURN
*****************
 5       CONTINUE
*****************
*
*********CALL MATVECTRANS( ONE, WORK(1,PTLD), ZERO, WORK(1,QTLD) )
*
         SCLR1 = ONE
         SCLR2 = ZERO
         NDX1 = ((PTLD - 1) * LDW) + 1
         NDX2 = ((QTLD - 1) * LDW) + 1
         RLBL = 6
         IJOB = 2
         RETURN
*****************
 6       CONTINUE
*****************
         TMPVAL = wzdotc( N, WORK(1,PTLD), 1, WORK(1,Q), 1 )
         IF (TMPVAL.EQ.0) THEN
*           Breakdown
            INFO = -11
            GO TO 20
         ENDIF
         ALPHA = RHO / TMPVAL
*
*        Compute current solution vector x.
*
         CALL zAXPY( N, ALPHA, WORK(1,P), 1, X, 1 )
*
*        Compute residual vector rk, find norm,
*        then check for tolerance.
*
         CALL zAXPY( N, -ALPHA, WORK(1,Q), 1, WORK(1,R), 1 )
*
*********RESID = dzNRM2( N, WORK(1,R), 1 ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 30
*
         NDX1 = NEED1
         NDX2 = NEED2
*        Prepare for resumption & return
         RLBL = 7
         IJOB = 6
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
*         CALL zAXPY( N, -ALPHA, WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
         CALL zAXPY( N, -conjg(ALPHA)
     $        , WORK(1,QTLD), 1, WORK(1,RTLD), 1 )
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
      INFO = -10
      RLBL = -1
      IJOB = -1
      RETURN
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
*     End of BICGREVCOM
*
      END
*     END SUBROUTINE zBICGREVCOM







