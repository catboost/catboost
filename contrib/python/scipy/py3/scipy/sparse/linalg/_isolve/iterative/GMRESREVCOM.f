*  -*- fortran -*-
      SUBROUTINE sGMRESREVCOM(N, B, X, RESTRT, WORK, LDW, WORK2,
     $                  LDW2, ITER, RESID, INFO, NDX1, NDX2, SCLR1, 
     $                  SCLR2, IJOB, TOL)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the
*     Solution of Linear Systems: Building Blocks for Iterative
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
*     EiITERkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, RESTRT, LDW, LDW2, ITER, INFO
      real  RESID, TOL
      INTEGER            NDX1, NDX2
      real   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      real   B( * ), X( * ), WORK( LDW,* ), WORK2( LDW2,* )
*     ..
*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method with preconditioning.
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
*          On input, the initial guess; on exit, the iterated solution.
*
*  RESTRT  (input) INTEGER
*          Restart parameter, .ls. = N. This parameter controls the amount
*          of memory required for matrix WORK2.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6+restrt).
*          Note that if the initial guess is the zero vector, then 
*          storing the initial residual is not necessary.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  WORK2   (workspace) DOUBLE PRECISION array, dimension (LDW2,2*RESTRT+2).
*          This workspace is used for constructing and storing the
*          upper Hessenberg matrix. The two extra columns are used to
*          store the Givens rotation matrices.
*
*  LDW2    (input) INTEGER
*          The leading dimension of the array WORK2.
*          LDW2 .gt. = max(2,RESTRT+1).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (output) DOUBLE PRECISION
*          On output, the norm of the preconditioned residual vector
*          if solution approximated to tolerance, otherwise reset
*          to input tolerance.
*
*  INFO    (output) INTEGER
*          =  0:  successful exit
*          =  1:  maximum number of iterations performed;
*                 convergence not achieved.
*            -5: Erroneous NDX1/NDX2 in INIT call.
*            -6: Erroneous RLBL.
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
*  TOL     (input) DOUBLE PRECISION. 
*          On input, the allowable absolute error tolerance
*          for the preconditioned residual.
*  ============================================================
*
*     .. Parameters ..
      real    ZERO, ONE
      PARAMETER         ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      INTEGER             OFSET
      PARAMETER         ( OFSET = 1000 )
*     ..
*     .. Local Scalars ..
      INTEGER             I, MAXIT, AV, GIV, H, R, S, V, W, Y,
     $                    NEED1, NEED2
      real  sdot
      real  toz
      real    TMPVAL
      real    RNORM, EPS,
     $     sNRM2,
     $     sAPPROXRES,
     $     sLAMCH

      LOGICAL BRKDWN
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL     sAXPY, sCOPY, sdot, sNRM2, sSCAL,
     $     sLAMCH
*     ..
*     .. Executable Statements ..
*
* Entry point, so test IJOB
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
         GOTO 200
      ENDIF
*
* init.
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      BRKDWN = .FALSE.
      EPS = sLAMCH('EPS')
*
*     Alias workspace columns.
*
      R   = 1
      S   = 2
      W   = 3
      Y   = 4
      AV  = 5
      V   = 6
*
      H   = 1
      GIV = H + RESTRT
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((S - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((W - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.V*OFSET ) .AND.
     $           ( NDX1.LE.V*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.GIV*OFSET ) .AND.
     $           ( NDX1.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((S - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((W - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.V*OFSET ) .AND.
     $           ( NDX2.LE.V*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.GIV*OFSET ) .AND.
     $           ( NDX2.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set initial residual.
*
      CALL sCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( sNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using X directly
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 1
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      ITER = 0
   10 CONTINUE
*
         ITER = ITER + 1
*
*        Construct the first column of V, and initialize S to the
*        elementary vector E1 scaled by RNORM.
*
*********CALL PSOLVE( WORK( 1,V ), WORK( 1,R ) )
*
         NDX1 = ((V - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
         RNORM = sNRM2( N, WORK( 1,V ), 1 )
         toz = ONE/RNORM
         CALL sSCAL( N, toz, WORK( 1,V ), 1 )
         TMPVAL = RNORM
         CALL sELEMVEC( 1, N, TMPVAL, WORK( 1,S ) )
*
*         DO 50 I = 1, RESTRT
         i = 1
         BRKDWN = .FALSE.
 49      if (i.gt.restrt) go to 50
************CALL MATVEC( ONE, WORK( 1,V+I-1 ), ZERO, WORK( 1,AV ) )
*
         NDX1 = ((V+I-1 - 1) * LDW) + 1
         NDX2 = ((AV    - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 3
         RETURN
*
*****************
 4       CONTINUE
*****************
*
*********CALL PSOLVE( WORK( 1,W ), WORK( 1,AV ) )
*
         NDX1 = ((W  - 1) * LDW) + 1
         NDX2 = ((AV - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*           Construct I-th column of H so that it is orthnormal to 
*           the previous I-1 columns.
*
            CALL sORTHOH( I, N, WORK2( 1,I+H-1 ), WORK( 1,V ), LDW,
     $                   WORK( 1,W ), BRKDWN, EPS )
*
            IF ( I.GT.0 )
*
*              Apply Givens rotations to the I-th column of H. This
*              effectively reduces the Hessenberg matrix to upper
*              triangular form during the RESTRT iterations.
*
     $         CALL sAPPLYGIVENS(I, WORK2( 1,I+H-1 ), WORK2( 1,GIV ),
     $                           LDW2 )
*
*           Approximate residual norm. Check tolerance. If okay, break out
*           from the inner loop.
*
            RESID = sAPPROXRES( I, WORK2( 1,I+H-1 ), WORK( 1,S ),
     $                         WORK2( 1,GIV ), LDW2 )
            IF ( RESID.LE.TOL .OR. BRKDWN ) THEN
               GO TO 51
            ENDIF
            i = i + 1
            go to 49
   50    CONTINUE
         i = restrt
*
*        Compute current solution vector X.
*
   51    CALL sUPDATE(I, N, X, WORK2( 1,H ), LDW2,
     $               WORK(1,Y), WORK( 1,S ), WORK( 1,V ), LDW )
*
*        Compute residual vector R, find norm,
*        then check for tolerance.
*
         CALL sCOPY( N, B, 1, WORK( 1,R ), 1 )
*********CALL MATVEC( -ONE, X, ONE, WORK( 1,R ) )
*
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = -ONE
         SCLR2 = ONE
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         WORK( I+1,S ) = sNRM2( N, WORK( 1,R ), 1 )
*
*********RESID = WORK( I+1,S ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 200
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
         IF( INFO.EQ.1 ) GO TO 200
         IF( BRKDWN ) THEN
*           Reached breakdown (= exact solution), but the external
*           tolerance check failed. Bail out with failure.
            INFO = 1
            GO TO 100
         ENDIF
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 100
         ENDIF
*
         GO TO 10
*
  100 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
  200 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1

      RETURN
*
*     End of GMRESREVCOM
*
      END
*     END SUBROUTINE sGMRESREVCOM
*
*     =========================================================
      SUBROUTINE sORTHOH( I, N, H, V, LDV, W, BRKDWN, EPS )
*
      IMPLICIT NONE
      INTEGER            I, N, LDV
      real   H( * ), W( * ), V( LDV,* )
      LOGICAL            BRKDWN
      real EPS
*
*     Construct the I-th column of the upper Hessenberg matrix H
*     using the Gram-Schmidt process on V and W.
*
      INTEGER            K
      real    sNRM2, ONE, H0, H1
      PARAMETER        ( ONE = 1.0D+0 )
      real    sdot
      real    TMPVAL
      EXTERNAL         sAXPY, sCOPY, sdot, sNRM2, sSCAL
*
      H0 = sNRM2( N, W, 1 )
      DO 10 K = 1, I
         H( K ) = sdot( N, V( 1,K ), 1, W, 1 )
         CALL sAXPY( N, -H( K ), V( 1,K ), 1, W, 1 )
   10 CONTINUE
      H1 = sNRM2( N, W, 1 )
      H( I+1 ) = H1
      CALL sCOPY( N, W, 1, V( 1,I+1 ), 1 )
      IF (.NOT.(H1.GT.EPS*H0)) THEN
*        Set to exactly 0: handled in UPDATE
         H( I+1 ) = 0d0
         BRKDWN = .TRUE.
      ELSE
         BRKDWN = .FALSE.
         TMPVAL = ONE / H( I+1 )
         CALL sSCAL( N, TMPVAL, V( 1,I+1 ), 1 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE sORTHOH
*     =========================================================
      SUBROUTINE sAPPLYGIVENS( I, H, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      real   H( * ), GIVENS( LDG,* )
*
*     This routine applies a sequence of I-1 Givens rotations to
*     the I-th column of H. The Givens parameters are stored, so that
*     the first I-2 Givens rotation matrices are known. The I-1st
*     Givens rotation is computed using BLAS 1 routine DROTG. Each
*     rotation is applied to the 2x1 vector [H( J ), H( J+1 )]',
*     which results in H( J+1 ) = 0.
*
      INTEGER            J
*      DOUBLE PRECISION   TEMP
      EXTERNAL           sROTG
*
*     .. Executable Statements ..
*
*     Construct I-1st rotation matrix.
*
*     CALL sROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL sGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
*     Apply 1,...,I-1st rotation matrices to the I-th column of H.
*
      DO 10 J = 1, I-1
         CALL sROTVEC(H( J ), H( J+1 ), GIVENS( J,1 ), GIVENS( J,2 ))
*        TEMP     =  GIVENS( J,1 ) * H( J ) + GIVENS( J,2 ) * H( J+1 ) 
*        H( J+1 ) = -GIVENS( J,2 ) * H( J ) + GIVENS( J,1 ) * H( J+1 )
*        H( J ) = TEMP
 10   CONTINUE
      call sgetgiv( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      call srotvec( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      RETURN
*
      END
*     END SUBROUTINE sAPPLYGIVENS
*
*     ===============================================================
      real
     $     FUNCTION sAPPROXRES( I, H, S, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      real   H( * ), S( * ), GIVENS( LDG,* )
*
*     This func allows the user to approximate the residual
*     using an updating scheme involving Givens rotations. The
*     rotation matrix is formed using [H( I ),H( I+1 )]' with the
*     intent of zeroing H( I+1 ), but here is applied to the 2x1
*     vector [S(I), S(I+1)]'.
*
      INTRINSIC          ABS
      EXTERNAL           sROTG
*
*     .. Executable Statements ..
*
*     CALL sROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL sGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      CALL sROTVEC( S( I ), S( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      sAPPROXRES = ABS( S( I+1 ) )
*
      RETURN
*
      END
*     END FUNCTION sAPPROXRES
*     ===============================================================
      SUBROUTINE sUPDATE( I, N, X, H, LDH, Y, S, V, LDV )
*
      IMPLICIT NONE
      INTEGER            N, I, J, LDH, LDV
      real   X( * ), Y( * ), S( * ), H( LDH,* ), V( LDV,* )
      EXTERNAL           sAXPY, sCOPY, sTRSV
*
*     Solve H*y = s for upper triangualar H.
*
      integer k, m
      CALL sCOPY( I, S, 1, Y, 1 )
*
*     Pseudoinverse vs. zero diagonals in H,
*     which may appear in breakdown conditions.
*
      J = I
 5    IF (J.GT.0) THEN
         IF (H(J,J).EQ.0) THEN
            Y(J) = 0
            J = J - 1
            GO TO 5
         ENDIF
      ENDIF
*
*     Solve triangular system
*
      IF (J.GT.0) THEN
         CALL sTRSV( 'UPPER', 'NOTRANS', 'NONUNIT', J, H, LDH, Y, 1 )
      END IF
*
*     Compute current solution vector X.
*
      DO 10 J = 1, I
         CALL sAXPY( N, Y( J ), V( 1,J ), 1, X, 1 )
   10 CONTINUE
*
      RETURN
*
      END
*     END SUBROUTINE sUPDATE
*
*     ===============================================================
      SUBROUTINE sGETGIV( A, B, C, S )
*
      IMPLICIT NONE
      real   A, B, C, S, TEMP, ZERO, ONE
      PARAMETER  ( 
     $     ZERO = 0.0, 
     $     ONE = 1.0 )
*
      IF ( ABS( B ).EQ.ZERO ) THEN
         C = ONE
         S = ZERO
      ELSE IF ( ABS( B ).GT.ABS( A ) ) THEN
         TEMP = -A / B
         S = ONE / SQRT( ONE + abs(TEMP)**2 )
         C = TEMP * S
*         S = b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = -a / SQRT( abs(a)**2 + abs(b)**2 )
      ELSE
         TEMP = -B / A
         C = ONE / SQRT( ONE + abs(TEMP)**2 )
         S = TEMP * C
*         S = -b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = a / SQRT( abs(a)**2 + abs(b)**2 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE sGETGIV
*
*     ================================================================
      SUBROUTINE sROTVEC( X, Y, C, S )
*
      IMPLICIT NONE
      real   X, Y, C, S, TEMP

*
      TEMP = (C) * X - (S) * Y
      Y    = S * X + C * Y
      X    = TEMP
*
      RETURN
*
      END
*     END SUBROUTINE sROTVEC
*
*     ===============================================================
      SUBROUTINE sELEMVEC( I, N, ALPHA, E )
*
*     Construct the I-th elementary vector E, scaled by ALPHA.
*
      IMPLICIT NONE
      INTEGER            I, J, N
      real   ALPHA, E( * )
*
*     .. Parameters ..
      real   ZERO
      PARAMETER        ( ZERO = 0.0D+0 )
*
      DO 10 J = 1, N
         E( J ) = ZERO
   10 CONTINUE
      E( I ) = ALPHA
*
      RETURN
*
      END
*     END SUBROUTINE sELEMVEC



      SUBROUTINE dGMRESREVCOM(N, B, X, RESTRT, WORK, LDW, WORK2,
     $                  LDW2, ITER, RESID, INFO, NDX1, NDX2, SCLR1, 
     $                  SCLR2, IJOB, TOL)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the
*     Solution of Linear Systems: Building Blocks for Iterative
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
*     EiITERkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, RESTRT, LDW, LDW2, ITER, INFO
      double precision  RESID, TOL
      INTEGER            NDX1, NDX2
      double precision   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      double precision   B( * ), X( * ), WORK( LDW,* ), WORK2( LDW2,* )
*     ..
*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method with preconditioning.
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
*          On input, the initial guess; on exit, the iterated solution.
*
*  RESTRT  (input) INTEGER
*          Restart parameter, .ls. = N. This parameter controls the amount
*          of memory required for matrix WORK2.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6+restrt).
*          Note that if the initial guess is the zero vector, then 
*          storing the initial residual is not necessary.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  WORK2   (workspace) DOUBLE PRECISION array, dimension (LDW2,2*RESTRT+2).
*          This workspace is used for constructing and storing the
*          upper Hessenberg matrix. The two extra columns are used to
*          store the Givens rotation matrices.
*
*  LDW2    (input) INTEGER
*          The leading dimension of the array WORK2.
*          LDW2 .gt. = max(2,RESTRT+1).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (output) DOUBLE PRECISION
*          On output, the norm of the preconditioned residual vector
*          if solution approximated to tolerance, otherwise reset
*          to input tolerance.
*
*  INFO    (output) INTEGER
*          =  0:  successful exit
*          =  1:  maximum number of iterations performed;
*                 convergence not achieved.
*            -5: Erroneous NDX1/NDX2 in INIT call.
*            -6: Erroneous RLBL.
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
*  TOL     (input) DOUBLE PRECISION. 
*          On input, the allowable absolute error tolerance
*          for the preconditioned residual.
*  ============================================================
*
*     .. Parameters ..
      double precision    ZERO, ONE
      PARAMETER         ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      INTEGER             OFSET
      PARAMETER         ( OFSET = 1000 )
*     ..
*     .. Local Scalars ..
      INTEGER             I, MAXIT, AV, GIV, H, R, S, V, W, Y,
     $                    NEED1, NEED2
      double precision  ddot
      double precision  toz
      double precision    TMPVAL
      double precision    RNORM, EPS,
     $     dNRM2,
     $     dAPPROXRES,
     $     dLAMCH

      LOGICAL BRKDWN
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL     dAXPY, dCOPY, ddot, dNRM2, dSCAL,
     $     dLAMCH
*     ..
*     .. Executable Statements ..
*
* Entry point, so test IJOB
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
         GOTO 200
      ENDIF
*
* init.
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      BRKDWN = .FALSE.
      EPS = dLAMCH('EPS')
*
*     Alias workspace columns.
*
      R   = 1
      S   = 2
      W   = 3
      Y   = 4
      AV  = 5
      V   = 6
*
      H   = 1
      GIV = H + RESTRT
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((S - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((W - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.V*OFSET ) .AND.
     $           ( NDX1.LE.V*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.GIV*OFSET ) .AND.
     $           ( NDX1.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((S - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((W - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.V*OFSET ) .AND.
     $           ( NDX2.LE.V*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.GIV*OFSET ) .AND.
     $           ( NDX2.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set initial residual.
*
      CALL dCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( dNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using X directly
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 1
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      ITER = 0
   10 CONTINUE
*
         ITER = ITER + 1
*
*        Construct the first column of V, and initialize S to the
*        elementary vector E1 scaled by RNORM.
*
*********CALL PSOLVE( WORK( 1,V ), WORK( 1,R ) )
*
         NDX1 = ((V - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
         RNORM = dNRM2( N, WORK( 1,V ), 1 )
         toz = ONE/RNORM
         CALL dSCAL( N, toz, WORK( 1,V ), 1 )
         TMPVAL = RNORM
         CALL dELEMVEC( 1, N, TMPVAL, WORK( 1,S ) )
*
*         DO 50 I = 1, RESTRT
         i = 1
         BRKDWN = .FALSE.
 49      if (i.gt.restrt) go to 50
************CALL MATVEC( ONE, WORK( 1,V+I-1 ), ZERO, WORK( 1,AV ) )
*
         NDX1 = ((V+I-1 - 1) * LDW) + 1
         NDX2 = ((AV    - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 3
         RETURN
*
*****************
 4       CONTINUE
*****************
*
*********CALL PSOLVE( WORK( 1,W ), WORK( 1,AV ) )
*
         NDX1 = ((W  - 1) * LDW) + 1
         NDX2 = ((AV - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*           Construct I-th column of H so that it is orthnormal to 
*           the previous I-1 columns.
*
            CALL dORTHOH( I, N, WORK2( 1,I+H-1 ), WORK( 1,V ), LDW,
     $                   WORK( 1,W ), BRKDWN, EPS )
*
            IF ( I.GT.0 )
*
*              Apply Givens rotations to the I-th column of H. This
*              effectively reduces the Hessenberg matrix to upper
*              triangular form during the RESTRT iterations.
*
     $         CALL dAPPLYGIVENS(I, WORK2( 1,I+H-1 ), WORK2( 1,GIV ),
     $                           LDW2 )
*
*           Approximate residual norm. Check tolerance. If okay, break out
*           from the inner loop.
*
            RESID = dAPPROXRES( I, WORK2( 1,I+H-1 ), WORK( 1,S ),
     $                         WORK2( 1,GIV ), LDW2 )
            IF ( RESID.LE.TOL .OR. BRKDWN ) THEN
               GO TO 51
            ENDIF
            i = i + 1
            go to 49
   50    CONTINUE
         i = restrt
*
*        Compute current solution vector X.
*
   51    CALL dUPDATE(I, N, X, WORK2( 1,H ), LDW2,
     $               WORK(1,Y), WORK( 1,S ), WORK( 1,V ), LDW )
*
*        Compute residual vector R, find norm,
*        then check for tolerance.
*
         CALL dCOPY( N, B, 1, WORK( 1,R ), 1 )
*********CALL MATVEC( -ONE, X, ONE, WORK( 1,R ) )
*
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = -ONE
         SCLR2 = ONE
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         WORK( I+1,S ) = dNRM2( N, WORK( 1,R ), 1 )
*
*********RESID = WORK( I+1,S ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 200
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
         IF( INFO.EQ.1 ) GO TO 200
         IF( BRKDWN ) THEN
*           Reached breakdown (= exact solution), but the external
*           tolerance check failed. Bail out with failure.
            INFO = 1
            GO TO 100
         ENDIF
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 100
         ENDIF
*
         GO TO 10
*
  100 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
  200 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1

      RETURN
*
*     End of GMRESREVCOM
*
      END
*     END SUBROUTINE dGMRESREVCOM
*
*     =========================================================
      SUBROUTINE dORTHOH( I, N, H, V, LDV, W, BRKDWN, EPS )
*
      IMPLICIT NONE
      INTEGER            I, N, LDV
      double precision   H( * ), W( * ), V( LDV,* )
      LOGICAL            BRKDWN
      double precision EPS
*
*     Construct the I-th column of the upper Hessenberg matrix H
*     using the Gram-Schmidt process on V and W.
*
      INTEGER            K
      double precision    dNRM2, ONE, H0, H1
      PARAMETER        ( ONE = 1.0D+0 )
      double precision    ddot
      double precision    TMPVAL
      EXTERNAL         dAXPY, dCOPY, ddot, dNRM2, dSCAL
*
      H0 = dNRM2( N, W, 1 )
      DO 10 K = 1, I
         H( K ) = ddot( N, V( 1,K ), 1, W, 1 )
         CALL dAXPY( N, -H( K ), V( 1,K ), 1, W, 1 )
   10 CONTINUE
      H1 = dNRM2( N, W, 1 )
      H( I+1 ) = H1
      CALL dCOPY( N, W, 1, V( 1,I+1 ), 1 )
      IF (.NOT.(H1.GT.EPS*H0)) THEN
*        Set to exactly 0: handled in UPDATE
         H( I+1 ) = 0d0
         BRKDWN = .TRUE.
      ELSE
         BRKDWN = .FALSE.
         TMPVAL = ONE / H( I+1 )
         CALL dSCAL( N, TMPVAL, V( 1,I+1 ), 1 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE dORTHOH
*     =========================================================
      SUBROUTINE dAPPLYGIVENS( I, H, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      double precision   H( * ), GIVENS( LDG,* )
*
*     This routine applies a sequence of I-1 Givens rotations to
*     the I-th column of H. The Givens parameters are stored, so that
*     the first I-2 Givens rotation matrices are known. The I-1st
*     Givens rotation is computed using BLAS 1 routine DROTG. Each
*     rotation is applied to the 2x1 vector [H( J ), H( J+1 )]',
*     which results in H( J+1 ) = 0.
*
      INTEGER            J
*      DOUBLE PRECISION   TEMP
      EXTERNAL           dROTG
*
*     .. Executable Statements ..
*
*     Construct I-1st rotation matrix.
*
*     CALL dROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL dGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
*     Apply 1,...,I-1st rotation matrices to the I-th column of H.
*
      DO 10 J = 1, I-1
         CALL dROTVEC(H( J ), H( J+1 ), GIVENS( J,1 ), GIVENS( J,2 ))
*        TEMP     =  GIVENS( J,1 ) * H( J ) + GIVENS( J,2 ) * H( J+1 ) 
*        H( J+1 ) = -GIVENS( J,2 ) * H( J ) + GIVENS( J,1 ) * H( J+1 )
*        H( J ) = TEMP
 10   CONTINUE
      call dgetgiv( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      call drotvec( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      RETURN
*
      END
*     END SUBROUTINE dAPPLYGIVENS
*
*     ===============================================================
      double precision
     $     FUNCTION dAPPROXRES( I, H, S, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      double precision   H( * ), S( * ), GIVENS( LDG,* )
*
*     This func allows the user to approximate the residual
*     using an updating scheme involving Givens rotations. The
*     rotation matrix is formed using [H( I ),H( I+1 )]' with the
*     intent of zeroing H( I+1 ), but here is applied to the 2x1
*     vector [S(I), S(I+1)]'.
*
      INTRINSIC          ABS
      EXTERNAL           dROTG
*
*     .. Executable Statements ..
*
*     CALL dROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL dGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      CALL dROTVEC( S( I ), S( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      dAPPROXRES = ABS( S( I+1 ) )
*
      RETURN
*
      END
*     END FUNCTION dAPPROXRES
*     ===============================================================
      SUBROUTINE dUPDATE( I, N, X, H, LDH, Y, S, V, LDV )
*
      IMPLICIT NONE
      INTEGER            N, I, J, LDH, LDV
      double precision   X( * ), Y( * ), S( * ), H( LDH,* ), V( LDV,* )
      EXTERNAL           dAXPY, dCOPY, dTRSV
*
*     Solve H*y = s for upper triangualar H.
*
      integer k, m
      CALL dCOPY( I, S, 1, Y, 1 )
*
*     Pseudoinverse vs. zero diagonals in H,
*     which may appear in breakdown conditions.
*
      J = I
 5    IF (J.GT.0) THEN
         IF (H(J,J).EQ.0) THEN
            Y(J) = 0
            J = J - 1
            GO TO 5
         ENDIF
      ENDIF
*
*     Solve triangular system
*
      IF (J.GT.0) THEN
         CALL dTRSV( 'UPPER', 'NOTRANS', 'NONUNIT', J, H, LDH, Y, 1 )
      END IF
*
*     Compute current solution vector X.
*
      DO 10 J = 1, I
         CALL dAXPY( N, Y( J ), V( 1,J ), 1, X, 1 )
   10 CONTINUE
*
      RETURN
*
      END
*     END SUBROUTINE dUPDATE
*
*     ===============================================================
      SUBROUTINE dGETGIV( A, B, C, S )
*
      IMPLICIT NONE
      double precision   A, B, C, S, TEMP, ZERO, ONE
      PARAMETER  ( 
     $     ZERO = 0.0, 
     $     ONE = 1.0 )
*
      IF ( ABS( B ).EQ.ZERO ) THEN
         C = ONE
         S = ZERO
      ELSE IF ( ABS( B ).GT.ABS( A ) ) THEN
         TEMP = -A / B
         S = ONE / SQRT( ONE + abs(TEMP)**2 )
         C = TEMP * S
*         S = b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = -a / SQRT( abs(a)**2 + abs(b)**2 )
      ELSE
         TEMP = -B / A
         C = ONE / SQRT( ONE + abs(TEMP)**2 )
         S = TEMP * C
*         S = -b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = a / SQRT( abs(a)**2 + abs(b)**2 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE dGETGIV
*
*     ================================================================
      SUBROUTINE dROTVEC( X, Y, C, S )
*
      IMPLICIT NONE
      double precision   X, Y, C, S, TEMP

*
      TEMP = (C) * X - (S) * Y
      Y    = S * X + C * Y
      X    = TEMP
*
      RETURN
*
      END
*     END SUBROUTINE dROTVEC
*
*     ===============================================================
      SUBROUTINE dELEMVEC( I, N, ALPHA, E )
*
*     Construct the I-th elementary vector E, scaled by ALPHA.
*
      IMPLICIT NONE
      INTEGER            I, J, N
      double precision   ALPHA, E( * )
*
*     .. Parameters ..
      double precision   ZERO
      PARAMETER        ( ZERO = 0.0D+0 )
*
      DO 10 J = 1, N
         E( J ) = ZERO
   10 CONTINUE
      E( I ) = ALPHA
*
      RETURN
*
      END
*     END SUBROUTINE dELEMVEC



      SUBROUTINE cGMRESREVCOM(N, B, X, RESTRT, WORK, LDW, WORK2,
     $                  LDW2, ITER, RESID, INFO, NDX1, NDX2, SCLR1, 
     $                  SCLR2, IJOB, TOL)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the
*     Solution of Linear Systems: Building Blocks for Iterative
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
*     EiITERkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, RESTRT, LDW, LDW2, ITER, INFO
      real  RESID, TOL
      INTEGER            NDX1, NDX2
      complex   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      complex   B( * ), X( * ), WORK( LDW,* ), WORK2( LDW2,* )
*     ..
*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method with preconditioning.
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
*          On input, the initial guess; on exit, the iterated solution.
*
*  RESTRT  (input) INTEGER
*          Restart parameter, .ls. = N. This parameter controls the amount
*          of memory required for matrix WORK2.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6+restrt).
*          Note that if the initial guess is the zero vector, then 
*          storing the initial residual is not necessary.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  WORK2   (workspace) DOUBLE PRECISION array, dimension (LDW2,2*RESTRT+2).
*          This workspace is used for constructing and storing the
*          upper Hessenberg matrix. The two extra columns are used to
*          store the Givens rotation matrices.
*
*  LDW2    (input) INTEGER
*          The leading dimension of the array WORK2.
*          LDW2 .gt. = max(2,RESTRT+1).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (output) DOUBLE PRECISION
*          On output, the norm of the preconditioned residual vector
*          if solution approximated to tolerance, otherwise reset
*          to input tolerance.
*
*  INFO    (output) INTEGER
*          =  0:  successful exit
*          =  1:  maximum number of iterations performed;
*                 convergence not achieved.
*            -5: Erroneous NDX1/NDX2 in INIT call.
*            -6: Erroneous RLBL.
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
*  TOL     (input) DOUBLE PRECISION. 
*          On input, the allowable absolute error tolerance
*          for the preconditioned residual.
*  ============================================================
*
*     .. Parameters ..
      real    ZERO, ONE
      PARAMETER         ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      INTEGER             OFSET
      PARAMETER         ( OFSET = 1000 )
*     ..
*     .. Local Scalars ..
      INTEGER             I, MAXIT, AV, GIV, H, R, S, V, W, Y,
     $                    NEED1, NEED2
      complex  wcdotc
      complex  toz
      complex    TMPVAL
      real    RNORM, EPS,
     $     scNRM2,
     $     scAPPROXRES,
     $     sLAMCH

      LOGICAL BRKDWN
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL     cAXPY, cCOPY, wcdotc, scNRM2, cSCAL,
     $     sLAMCH
*     ..
*     .. Executable Statements ..
*
* Entry point, so test IJOB
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
         GOTO 200
      ENDIF
*
* init.
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      BRKDWN = .FALSE.
      EPS = sLAMCH('EPS')
*
*     Alias workspace columns.
*
      R   = 1
      S   = 2
      W   = 3
      Y   = 4
      AV  = 5
      V   = 6
*
      H   = 1
      GIV = H + RESTRT
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((S - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((W - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.V*OFSET ) .AND.
     $           ( NDX1.LE.V*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.GIV*OFSET ) .AND.
     $           ( NDX1.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((S - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((W - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.V*OFSET ) .AND.
     $           ( NDX2.LE.V*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.GIV*OFSET ) .AND.
     $           ( NDX2.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set initial residual.
*
      CALL cCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( scNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using X directly
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 1
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      ITER = 0
   10 CONTINUE
*
         ITER = ITER + 1
*
*        Construct the first column of V, and initialize S to the
*        elementary vector E1 scaled by RNORM.
*
*********CALL PSOLVE( WORK( 1,V ), WORK( 1,R ) )
*
         NDX1 = ((V - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
         RNORM = scNRM2( N, WORK( 1,V ), 1 )
         toz = ONE/RNORM
         CALL cSCAL( N, toz, WORK( 1,V ), 1 )
         TMPVAL = RNORM
         CALL cELEMVEC( 1, N, TMPVAL, WORK( 1,S ) )
*
*         DO 50 I = 1, RESTRT
         i = 1
         BRKDWN = .FALSE.
 49      if (i.gt.restrt) go to 50
************CALL MATVEC( ONE, WORK( 1,V+I-1 ), ZERO, WORK( 1,AV ) )
*
         NDX1 = ((V+I-1 - 1) * LDW) + 1
         NDX2 = ((AV    - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 3
         RETURN
*
*****************
 4       CONTINUE
*****************
*
*********CALL PSOLVE( WORK( 1,W ), WORK( 1,AV ) )
*
         NDX1 = ((W  - 1) * LDW) + 1
         NDX2 = ((AV - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*           Construct I-th column of H so that it is orthnormal to 
*           the previous I-1 columns.
*
            CALL cORTHOH( I, N, WORK2( 1,I+H-1 ), WORK( 1,V ), LDW,
     $                   WORK( 1,W ), BRKDWN, EPS )
*
            IF ( I.GT.0 )
*
*              Apply Givens rotations to the I-th column of H. This
*              effectively reduces the Hessenberg matrix to upper
*              triangular form during the RESTRT iterations.
*
     $         CALL cAPPLYGIVENS(I, WORK2( 1,I+H-1 ), WORK2( 1,GIV ),
     $                           LDW2 )
*
*           Approximate residual norm. Check tolerance. If okay, break out
*           from the inner loop.
*
            RESID = scAPPROXRES( I, WORK2( 1,I+H-1 ), WORK( 1,S ),
     $                         WORK2( 1,GIV ), LDW2 )
            IF ( RESID.LE.TOL .OR. BRKDWN ) THEN
               GO TO 51
            ENDIF
            i = i + 1
            go to 49
   50    CONTINUE
         i = restrt
*
*        Compute current solution vector X.
*
   51    CALL cUPDATE(I, N, X, WORK2( 1,H ), LDW2,
     $               WORK(1,Y), WORK( 1,S ), WORK( 1,V ), LDW )
*
*        Compute residual vector R, find norm,
*        then check for tolerance.
*
         CALL cCOPY( N, B, 1, WORK( 1,R ), 1 )
*********CALL MATVEC( -ONE, X, ONE, WORK( 1,R ) )
*
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = -ONE
         SCLR2 = ONE
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         WORK( I+1,S ) = scNRM2( N, WORK( 1,R ), 1 )
*
*********RESID = WORK( I+1,S ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 200
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
         IF( INFO.EQ.1 ) GO TO 200
         IF( BRKDWN ) THEN
*           Reached breakdown (= exact solution), but the external
*           tolerance check failed. Bail out with failure.
            INFO = 1
            GO TO 100
         ENDIF
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 100
         ENDIF
*
         GO TO 10
*
  100 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
  200 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1

      RETURN
*
*     End of GMRESREVCOM
*
      END
*     END SUBROUTINE cGMRESREVCOM
*
*     =========================================================
      SUBROUTINE cORTHOH( I, N, H, V, LDV, W, BRKDWN, EPS )
*
      IMPLICIT NONE
      INTEGER            I, N, LDV
      complex   H( * ), W( * ), V( LDV,* )
      LOGICAL            BRKDWN
      real EPS
*
*     Construct the I-th column of the upper Hessenberg matrix H
*     using the Gram-Schmidt process on V and W.
*
      INTEGER            K
      real    scNRM2, ONE, H0, H1
      PARAMETER        ( ONE = 1.0D+0 )
      complex    wcdotc
      complex    TMPVAL
      EXTERNAL         cAXPY, cCOPY, wcdotc, scNRM2, cSCAL
*
      H0 = scNRM2( N, W, 1 )
      DO 10 K = 1, I
         H( K ) = wcdotc( N, V( 1,K ), 1, W, 1 )
         CALL cAXPY( N, -H( K ), V( 1,K ), 1, W, 1 )
   10 CONTINUE
      H1 = scNRM2( N, W, 1 )
      H( I+1 ) = H1
      CALL cCOPY( N, W, 1, V( 1,I+1 ), 1 )
      IF (.NOT.(H1.GT.EPS*H0)) THEN
*        Set to exactly 0: handled in UPDATE
         H( I+1 ) = 0d0
         BRKDWN = .TRUE.
      ELSE
         BRKDWN = .FALSE.
         TMPVAL = ONE / H( I+1 )
         CALL cSCAL( N, TMPVAL, V( 1,I+1 ), 1 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE cORTHOH
*     =========================================================
      SUBROUTINE cAPPLYGIVENS( I, H, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      complex   H( * ), GIVENS( LDG,* )
*
*     This routine applies a sequence of I-1 Givens rotations to
*     the I-th column of H. The Givens parameters are stored, so that
*     the first I-2 Givens rotation matrices are known. The I-1st
*     Givens rotation is computed using BLAS 1 routine DROTG. Each
*     rotation is applied to the 2x1 vector [H( J ), H( J+1 )]',
*     which results in H( J+1 ) = 0.
*
      INTEGER            J
*      DOUBLE PRECISION   TEMP
      EXTERNAL           cROTG
*
*     .. Executable Statements ..
*
*     Construct I-1st rotation matrix.
*
*     CALL cROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL cGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
*     Apply 1,...,I-1st rotation matrices to the I-th column of H.
*
      DO 10 J = 1, I-1
         CALL cROTVEC(H( J ), H( J+1 ), GIVENS( J,1 ), GIVENS( J,2 ))
*        TEMP     =  GIVENS( J,1 ) * H( J ) + GIVENS( J,2 ) * H( J+1 ) 
*        H( J+1 ) = -GIVENS( J,2 ) * H( J ) + GIVENS( J,1 ) * H( J+1 )
*        H( J ) = TEMP
 10   CONTINUE
      call cgetgiv( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      call crotvec( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      RETURN
*
      END
*     END SUBROUTINE cAPPLYGIVENS
*
*     ===============================================================
      real
     $     FUNCTION scAPPROXRES( I, H, S, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      complex   H( * ), S( * ), GIVENS( LDG,* )
*
*     This func allows the user to approximate the residual
*     using an updating scheme involving Givens rotations. The
*     rotation matrix is formed using [H( I ),H( I+1 )]' with the
*     intent of zeroing H( I+1 ), but here is applied to the 2x1
*     vector [S(I), S(I+1)]'.
*
      INTRINSIC          ABS
      EXTERNAL           cROTG
*
*     .. Executable Statements ..
*
*     CALL cROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL cGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      CALL cROTVEC( S( I ), S( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      scAPPROXRES = ABS( S( I+1 ) )
*
      RETURN
*
      END
*     END FUNCTION scAPPROXRES
*     ===============================================================
      SUBROUTINE cUPDATE( I, N, X, H, LDH, Y, S, V, LDV )
*
      IMPLICIT NONE
      INTEGER            N, I, J, LDH, LDV
      complex   X( * ), Y( * ), S( * ), H( LDH,* ), V( LDV,* )
      EXTERNAL           cAXPY, cCOPY, cTRSV
*
*     Solve H*y = s for upper triangualar H.
*
      integer k, m
      CALL cCOPY( I, S, 1, Y, 1 )
*
*     Pseudoinverse vs. zero diagonals in H,
*     which may appear in breakdown conditions.
*
      J = I
 5    IF (J.GT.0) THEN
         IF (H(J,J).EQ.0) THEN
            Y(J) = 0
            J = J - 1
            GO TO 5
         ENDIF
      ENDIF
*
*     Solve triangular system
*
      IF (J.GT.0) THEN
         CALL cTRSV( 'UPPER', 'NOTRANS', 'NONUNIT', J, H, LDH, Y, 1 )
      END IF
*
*     Compute current solution vector X.
*
      DO 10 J = 1, I
         CALL cAXPY( N, Y( J ), V( 1,J ), 1, X, 1 )
   10 CONTINUE
*
      RETURN
*
      END
*     END SUBROUTINE cUPDATE
*
*     ===============================================================
      SUBROUTINE cGETGIV( A, B, C, S )
*
      IMPLICIT NONE
      complex   A, B, C, S, TEMP, ZERO, ONE
      PARAMETER  ( 
     $     ZERO = 0.0, 
     $     ONE = 1.0 )
*
      IF ( ABS( B ).EQ.ZERO ) THEN
         C = ONE
         S = ZERO
      ELSE IF ( ABS( B ).GT.ABS( A ) ) THEN
         TEMP = -A / B
         S = ONE / SQRT( ONE + abs(TEMP)**2 )
         C = TEMP * S
*         S = b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = -a / SQRT( abs(a)**2 + abs(b)**2 )
      ELSE
         TEMP = -B / A
         C = ONE / SQRT( ONE + abs(TEMP)**2 )
         S = TEMP * C
*         S = -b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = a / SQRT( abs(a)**2 + abs(b)**2 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE cGETGIV
*
*     ================================================================
      SUBROUTINE cROTVEC( X, Y, C, S )
*
      IMPLICIT NONE
      complex   X, Y, C, S, TEMP

*
      TEMP = conjg(C) * X - conjg(S) * Y
      Y    = S * X + C * Y
      X    = TEMP
*
      RETURN
*
      END
*     END SUBROUTINE cROTVEC
*
*     ===============================================================
      SUBROUTINE cELEMVEC( I, N, ALPHA, E )
*
*     Construct the I-th elementary vector E, scaled by ALPHA.
*
      IMPLICIT NONE
      INTEGER            I, J, N
      complex   ALPHA, E( * )
*
*     .. Parameters ..
      real   ZERO
      PARAMETER        ( ZERO = 0.0D+0 )
*
      DO 10 J = 1, N
         E( J ) = ZERO
   10 CONTINUE
      E( I ) = ALPHA
*
      RETURN
*
      END
*     END SUBROUTINE cELEMVEC



      SUBROUTINE zGMRESREVCOM(N, B, X, RESTRT, WORK, LDW, WORK2,
     $                  LDW2, ITER, RESID, INFO, NDX1, NDX2, SCLR1, 
     $                  SCLR2, IJOB, TOL)
*
*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the
*     Solution of Linear Systems: Building Blocks for Iterative
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
*     EiITERkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            N, RESTRT, LDW, LDW2, ITER, INFO
      double precision  RESID, TOL
      INTEGER            NDX1, NDX2
      double complex   SCLR1, SCLR2
      INTEGER            IJOB
*     ..
*     .. Array Arguments ..
      double complex   B( * ), X( * ), WORK( LDW,* ), WORK2( LDW2,* )
*     ..
*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method with preconditioning.
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
*          On input, the initial guess; on exit, the iterated solution.
*
*  RESTRT  (input) INTEGER
*          Restart parameter, .ls. = N. This parameter controls the amount
*          of memory required for matrix WORK2.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,6+restrt).
*          Note that if the initial guess is the zero vector, then 
*          storing the initial residual is not necessary.
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW .gt. = max(1,N).
*
*  WORK2   (workspace) DOUBLE PRECISION array, dimension (LDW2,2*RESTRT+2).
*          This workspace is used for constructing and storing the
*          upper Hessenberg matrix. The two extra columns are used to
*          store the Givens rotation matrices.
*
*  LDW2    (input) INTEGER
*          The leading dimension of the array WORK2.
*          LDW2 .gt. = max(2,RESTRT+1).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (output) DOUBLE PRECISION
*          On output, the norm of the preconditioned residual vector
*          if solution approximated to tolerance, otherwise reset
*          to input tolerance.
*
*  INFO    (output) INTEGER
*          =  0:  successful exit
*          =  1:  maximum number of iterations performed;
*                 convergence not achieved.
*            -5: Erroneous NDX1/NDX2 in INIT call.
*            -6: Erroneous RLBL.
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
*  TOL     (input) DOUBLE PRECISION. 
*          On input, the allowable absolute error tolerance
*          for the preconditioned residual.
*  ============================================================
*
*     .. Parameters ..
      double precision    ZERO, ONE
      PARAMETER         ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      INTEGER             OFSET
      PARAMETER         ( OFSET = 1000 )
*     ..
*     .. Local Scalars ..
      INTEGER             I, MAXIT, AV, GIV, H, R, S, V, W, Y,
     $                    NEED1, NEED2
      double complex  wzdotc
      double complex  toz
      double complex    TMPVAL
      double precision    RNORM, EPS,
     $     dzNRM2,
     $     dzAPPROXRES,
     $     dLAMCH

      LOGICAL BRKDWN
*
*     indicates where to resume from. Only valid when IJOB = 2!
      INTEGER RLBL
*
*     saving all.
      SAVE
*
*     ..
*     .. External Routines ..
      EXTERNAL     zAXPY, zCOPY, wzdotc, dzNRM2, zSCAL,
     $     dLAMCH
*     ..
*     .. Executable Statements ..
*
* Entry point, so test IJOB
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
         GOTO 200
      ENDIF
*
* init.
*****************
 1    CONTINUE
*****************
*
      INFO = 0
      MAXIT = ITER
      BRKDWN = .FALSE.
      EPS = dLAMCH('EPS')
*
*     Alias workspace columns.
*
      R   = 1
      S   = 2
      W   = 3
      Y   = 4
      AV  = 5
      V   = 6
*
      H   = 1
      GIV = H + RESTRT
*
*     Check if caller will need indexing info.
*
      IF( NDX1.NE.-1 ) THEN
         IF( NDX1.EQ.1 ) THEN
            NEED1 = ((R - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.2 ) THEN
            NEED1 = ((S - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.3 ) THEN
            NEED1 = ((W - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.4 ) THEN
            NEED1 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.5 ) THEN
            NEED1 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX1.EQ.6 ) THEN
            NEED1 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.V*OFSET ) .AND.
     $           ( NDX1.LE.V*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX1.GT.GIV*OFSET ) .AND.
     $           ( NDX1.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED1 = ((NDX1-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED1 = NDX1
      ENDIF
*
      IF( NDX2.NE.-1 ) THEN
         IF( NDX2.EQ.1 ) THEN
            NEED2 = ((R - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.2 ) THEN
            NEED2 = ((S - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.3 ) THEN
            NEED2 = ((W - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.4 ) THEN
            NEED2 = ((Y - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.5 ) THEN
            NEED2 = ((AV - 1) * LDW) + 1
         ELSEIF( NDX2.EQ.6 ) THEN
            NEED2 = ((V - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.V*OFSET ) .AND.
     $           ( NDX2.LE.V*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-V*OFSET - 1) * LDW) + 1
         ELSEIF( ( NDX2.GT.GIV*OFSET ) .AND.
     $           ( NDX2.LE.GIV*OFSET+RESTRT ) ) THEN
            NEED2 = ((NDX2-GIV*OFSET - 1) * LDW) + 1
         ELSE
*           report error
            INFO = -5
            GO TO 100
         ENDIF
      ELSE
         NEED2 = NDX2
      ENDIF
*
*     Set initial residual.
*
      CALL zCOPY( N, B, 1, WORK(1,R), 1 )
      IF ( dzNRM2( N, X, 1 ).NE.ZERO ) THEN
*********CALL MATVEC( -ONE, X, ONE, WORK(1,R) )
*        Note: using X directly
         SCLR1 = -ONE
         SCLR2 = ONE
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*
*        Prepare for resumption & return
         RLBL = 2
         IJOB = 1
         RETURN
      ENDIF
*
*****************
 2    CONTINUE
*****************
*
      ITER = 0
   10 CONTINUE
*
         ITER = ITER + 1
*
*        Construct the first column of V, and initialize S to the
*        elementary vector E1 scaled by RNORM.
*
*********CALL PSOLVE( WORK( 1,V ), WORK( 1,R ) )
*
         NDX1 = ((V - 1) * LDW) + 1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 3
         IJOB = 2
         RETURN
*
*****************
 3       CONTINUE
*****************
*
         RNORM = dzNRM2( N, WORK( 1,V ), 1 )
         toz = ONE/RNORM
         CALL zSCAL( N, toz, WORK( 1,V ), 1 )
         TMPVAL = RNORM
         CALL zELEMVEC( 1, N, TMPVAL, WORK( 1,S ) )
*
*         DO 50 I = 1, RESTRT
         i = 1
         BRKDWN = .FALSE.
 49      if (i.gt.restrt) go to 50
************CALL MATVEC( ONE, WORK( 1,V+I-1 ), ZERO, WORK( 1,AV ) )
*
         NDX1 = ((V+I-1 - 1) * LDW) + 1
         NDX2 = ((AV    - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = ONE
         SCLR2 = ZERO
         RLBL = 4
         IJOB = 3
         RETURN
*
*****************
 4       CONTINUE
*****************
*
*********CALL PSOLVE( WORK( 1,W ), WORK( 1,AV ) )
*
         NDX1 = ((W  - 1) * LDW) + 1
         NDX2 = ((AV - 1) * LDW) + 1
*        Prepare for return & return
         RLBL = 5
         IJOB = 2
         RETURN
*
*****************
 5       CONTINUE
*****************
*
*           Construct I-th column of H so that it is orthnormal to 
*           the previous I-1 columns.
*
            CALL zORTHOH( I, N, WORK2( 1,I+H-1 ), WORK( 1,V ), LDW,
     $                   WORK( 1,W ), BRKDWN, EPS )
*
            IF ( I.GT.0 )
*
*              Apply Givens rotations to the I-th column of H. This
*              effectively reduces the Hessenberg matrix to upper
*              triangular form during the RESTRT iterations.
*
     $         CALL zAPPLYGIVENS(I, WORK2( 1,I+H-1 ), WORK2( 1,GIV ),
     $                           LDW2 )
*
*           Approximate residual norm. Check tolerance. If okay, break out
*           from the inner loop.
*
            RESID = dzAPPROXRES( I, WORK2( 1,I+H-1 ), WORK( 1,S ),
     $                         WORK2( 1,GIV ), LDW2 )
            IF ( RESID.LE.TOL .OR. BRKDWN ) THEN
               GO TO 51
            ENDIF
            i = i + 1
            go to 49
   50    CONTINUE
         i = restrt
*
*        Compute current solution vector X.
*
   51    CALL zUPDATE(I, N, X, WORK2( 1,H ), LDW2,
     $               WORK(1,Y), WORK( 1,S ), WORK( 1,V ), LDW )
*
*        Compute residual vector R, find norm,
*        then check for tolerance.
*
         CALL zCOPY( N, B, 1, WORK( 1,R ), 1 )
*********CALL MATVEC( -ONE, X, ONE, WORK( 1,R ) )
*
         NDX1 = -1
         NDX2 = ((R - 1) * LDW) + 1
*        Prepare for return & return
         SCLR1 = -ONE
         SCLR2 = ONE
         RLBL = 6
         IJOB = 1
         RETURN
*
*****************
 6       CONTINUE
*****************
*
         WORK( I+1,S ) = dzNRM2( N, WORK( 1,R ), 1 )
*
*********RESID = WORK( I+1,S ) / BNRM2
*********IF ( RESID.LE.TOL  ) GO TO 200
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
         IF( INFO.EQ.1 ) GO TO 200
         IF( BRKDWN ) THEN
*           Reached breakdown (= exact solution), but the external
*           tolerance check failed. Bail out with failure.
            INFO = 1
            GO TO 100
         ENDIF
*
         IF ( ITER.EQ.MAXIT ) THEN
            INFO = 1
            GO TO 100
         ENDIF
*
         GO TO 10
*
  100 CONTINUE
*
*     Iteration fails.
*
      RLBL = -1
      IJOB = -1
      RETURN
*
  200 CONTINUE
*
*     Iteration successful; return.
*
      INFO = 0
      RLBL = -1
      IJOB = -1

      RETURN
*
*     End of GMRESREVCOM
*
      END
*     END SUBROUTINE zGMRESREVCOM
*
*     =========================================================
      SUBROUTINE zORTHOH( I, N, H, V, LDV, W, BRKDWN, EPS )
*
      IMPLICIT NONE
      INTEGER            I, N, LDV
      double complex   H( * ), W( * ), V( LDV,* )
      LOGICAL            BRKDWN
      double precision EPS
*
*     Construct the I-th column of the upper Hessenberg matrix H
*     using the Gram-Schmidt process on V and W.
*
      INTEGER            K
      double precision    dzNRM2, ONE, H0, H1
      PARAMETER        ( ONE = 1.0D+0 )
      double complex    wzdotc
      double complex    TMPVAL
      EXTERNAL         zAXPY, zCOPY, wzdotc, dzNRM2, zSCAL
*
      H0 = dzNRM2( N, W, 1 )
      DO 10 K = 1, I
         H( K ) = wzdotc( N, V( 1,K ), 1, W, 1 )
         CALL zAXPY( N, -H( K ), V( 1,K ), 1, W, 1 )
   10 CONTINUE
      H1 = dzNRM2( N, W, 1 )
      H( I+1 ) = H1
      CALL zCOPY( N, W, 1, V( 1,I+1 ), 1 )
      IF (.NOT.(H1.GT.EPS*H0)) THEN
*        Set to exactly 0: handled in UPDATE
         H( I+1 ) = 0d0
         BRKDWN = .TRUE.
      ELSE
         BRKDWN = .FALSE.
         TMPVAL = ONE / H( I+1 )
         CALL zSCAL( N, TMPVAL, V( 1,I+1 ), 1 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE zORTHOH
*     =========================================================
      SUBROUTINE zAPPLYGIVENS( I, H, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      double complex   H( * ), GIVENS( LDG,* )
*
*     This routine applies a sequence of I-1 Givens rotations to
*     the I-th column of H. The Givens parameters are stored, so that
*     the first I-2 Givens rotation matrices are known. The I-1st
*     Givens rotation is computed using BLAS 1 routine DROTG. Each
*     rotation is applied to the 2x1 vector [H( J ), H( J+1 )]',
*     which results in H( J+1 ) = 0.
*
      INTEGER            J
*      DOUBLE PRECISION   TEMP
      EXTERNAL           zROTG
*
*     .. Executable Statements ..
*
*     Construct I-1st rotation matrix.
*
*     CALL zROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL zGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
*     Apply 1,...,I-1st rotation matrices to the I-th column of H.
*
      DO 10 J = 1, I-1
         CALL zROTVEC(H( J ), H( J+1 ), GIVENS( J,1 ), GIVENS( J,2 ))
*        TEMP     =  GIVENS( J,1 ) * H( J ) + GIVENS( J,2 ) * H( J+1 ) 
*        H( J+1 ) = -GIVENS( J,2 ) * H( J ) + GIVENS( J,1 ) * H( J+1 )
*        H( J ) = TEMP
 10   CONTINUE
      call zgetgiv( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      call zrotvec( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      RETURN
*
      END
*     END SUBROUTINE zAPPLYGIVENS
*
*     ===============================================================
      double precision
     $     FUNCTION dzAPPROXRES( I, H, S, GIVENS, LDG )
*
      IMPLICIT NONE
      INTEGER            I, LDG
      double complex   H( * ), S( * ), GIVENS( LDG,* )
*
*     This func allows the user to approximate the residual
*     using an updating scheme involving Givens rotations. The
*     rotation matrix is formed using [H( I ),H( I+1 )]' with the
*     intent of zeroing H( I+1 ), but here is applied to the 2x1
*     vector [S(I), S(I+1)]'.
*
      INTRINSIC          ABS
      EXTERNAL           zROTG
*
*     .. Executable Statements ..
*
*     CALL zROTG( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*      CALL zGETGIV( H( I ), H( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
      CALL zROTVEC( S( I ), S( I+1 ), GIVENS( I,1 ), GIVENS( I,2 ) )
*
      dzAPPROXRES = ABS( S( I+1 ) )
*
      RETURN
*
      END
*     END FUNCTION dzAPPROXRES
*     ===============================================================
      SUBROUTINE zUPDATE( I, N, X, H, LDH, Y, S, V, LDV )
*
      IMPLICIT NONE
      INTEGER            N, I, J, LDH, LDV
      double complex   X( * ), Y( * ), S( * ), H( LDH,* ), V( LDV,* )
      EXTERNAL           zAXPY, zCOPY, zTRSV
*
*     Solve H*y = s for upper triangualar H.
*
      integer k, m
      CALL zCOPY( I, S, 1, Y, 1 )
*
*     Pseudoinverse vs. zero diagonals in H,
*     which may appear in breakdown conditions.
*
      J = I
 5    IF (J.GT.0) THEN
         IF (H(J,J).EQ.0) THEN
            Y(J) = 0
            J = J - 1
            GO TO 5
         ENDIF
      ENDIF
*
*     Solve triangular system
*
      IF (J.GT.0) THEN
         CALL zTRSV( 'UPPER', 'NOTRANS', 'NONUNIT', J, H, LDH, Y, 1 )
      END IF
*
*     Compute current solution vector X.
*
      DO 10 J = 1, I
         CALL zAXPY( N, Y( J ), V( 1,J ), 1, X, 1 )
   10 CONTINUE
*
      RETURN
*
      END
*     END SUBROUTINE zUPDATE
*
*     ===============================================================
      SUBROUTINE zGETGIV( A, B, C, S )
*
      IMPLICIT NONE
      double complex   A, B, C, S, TEMP, ZERO, ONE
      PARAMETER  ( 
     $     ZERO = 0.0, 
     $     ONE = 1.0 )
*
      IF ( ABS( B ).EQ.ZERO ) THEN
         C = ONE
         S = ZERO
      ELSE IF ( ABS( B ).GT.ABS( A ) ) THEN
         TEMP = -A / B
         S = ONE / SQRT( ONE + abs(TEMP)**2 )
         C = TEMP * S
*         S = b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = -a / SQRT( abs(a)**2 + abs(b)**2 )
      ELSE
         TEMP = -B / A
         C = ONE / SQRT( ONE + abs(TEMP)**2 )
         S = TEMP * C
*         S = -b / SQRT( abs(a)**2 + abs(b)**2 )
*         C = a / SQRT( abs(a)**2 + abs(b)**2 )
      ENDIF
*
      RETURN
*
      END
*     END SUBROUTINE zGETGIV
*
*     ================================================================
      SUBROUTINE zROTVEC( X, Y, C, S )
*
      IMPLICIT NONE
      double complex   X, Y, C, S, TEMP

*
      TEMP = conjg(C) * X - conjg(S) * Y
      Y    = S * X + C * Y
      X    = TEMP
*
      RETURN
*
      END
*     END SUBROUTINE zROTVEC
*
*     ===============================================================
      SUBROUTINE zELEMVEC( I, N, ALPHA, E )
*
*     Construct the I-th elementary vector E, scaled by ALPHA.
*
      IMPLICIT NONE
      INTEGER            I, J, N
      double complex   ALPHA, E( * )
*
*     .. Parameters ..
      double precision   ZERO
      PARAMETER        ( ZERO = 0.0D+0 )
*
      DO 10 J = 1, N
         E( J ) = ZERO
   10 CONTINUE
      E( I ) = ALPHA
*
      RETURN
*
      END
*     END SUBROUTINE zELEMVEC



