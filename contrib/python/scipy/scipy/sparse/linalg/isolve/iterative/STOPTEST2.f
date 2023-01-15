* -*- fortran -*-
C     STOPTEST2

*  Purpose
*  =======
*
*  Computes the stopping criterion 2.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER.
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
*
*  INFO    (output) INTEGER
*          On exit, 1/0 depending on whether stopping criterion
*          was met or not.
*
*  BNRM2   (input/output) DOUBLE PRECISION.
*          On first time entry, will be -1.0.
*          On first time exit will contain norm2(B)
*          On all subsequent entry/exit's unchanged.
*
*  RESID   (output) DOUBLE PRECISION.
*          On exit, the computed stopping measure.
*
*  TOL     (input) DOUBLE PRECISION.
*          On input, the allowable convergence measure.
*
*  R       (input) DOUBLE PRECISION array, dimension N.
*          On entry, the residual.
*          Unchanged on exit.
*
*  B       (input) DOUBLE PRECISION array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  BLAS CALLS:   DNRM2
*  ============================================================
*

      SUBROUTINE sSTOPTEST2( N, R, B, BNRM2, RESID, TOL, INFO )
      INTEGER            N, INFO
      real RESID, TOL, BNRM2
      real   R( * ), B( * )
      real   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      real wsNRM2
      EXTERNAL         wsNRM2
      IF( INFO.EQ.-1 ) THEN
         BNRM2 = wsNRM2( N, B, 1 )
         IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
      ENDIF
      RESID = wsNRM2( N, R, 1 ) / BNRM2
      INFO = 0
      IF ( RESID.LE.TOL )
     $     INFO = 1
      RETURN
      END
*     END SUBROUTINE sSTOPTEST2
 



      SUBROUTINE dSTOPTEST2( N, R, B, BNRM2, RESID, TOL, INFO )
      INTEGER            N, INFO
      double precision RESID, TOL, BNRM2
      double precision   R( * ), B( * )
      double precision   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      double precision dNRM2
      EXTERNAL         dNRM2
      IF( INFO.EQ.-1 ) THEN
         BNRM2 = dNRM2( N, B, 1 )
         IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
      ENDIF
      RESID = dNRM2( N, R, 1 ) / BNRM2
      INFO = 0
      IF ( RESID.LE.TOL )
     $     INFO = 1
      RETURN
      END
*     END SUBROUTINE dSTOPTEST2
 



      SUBROUTINE cSTOPTEST2( N, R, B, BNRM2, RESID, TOL, INFO )
      INTEGER            N, INFO
      real RESID, TOL, BNRM2
      complex   R( * ), B( * )
      real   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      real wscNRM2
      EXTERNAL         wscNRM2
      IF( INFO.EQ.-1 ) THEN
         BNRM2 = wscNRM2( N, B, 1 )
         IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
      ENDIF
      RESID = wscNRM2( N, R, 1 ) / BNRM2
      INFO = 0
      IF ( RESID.LE.TOL )
     $     INFO = 1
      RETURN
      END
*     END SUBROUTINE cSTOPTEST2
 



      SUBROUTINE zSTOPTEST2( N, R, B, BNRM2, RESID, TOL, INFO )
      INTEGER            N, INFO
      double precision RESID, TOL, BNRM2
      double complex   R( * ), B( * )
      double precision   ZERO, ONE
      PARAMETER        ( ZERO = 0.0D+0, ONE = 1.0D+0 )
      double precision dzNRM2
      EXTERNAL         dzNRM2
      IF( INFO.EQ.-1 ) THEN
         BNRM2 = dzNRM2( N, B, 1 )
         IF ( BNRM2.EQ.ZERO ) BNRM2 = ONE
      ENDIF
      RESID = dzNRM2( N, R, 1 ) / BNRM2
      INFO = 0
      IF ( RESID.LE.TOL )
     $     INFO = 1
      RETURN
      END
*     END SUBROUTINE zSTOPTEST2
 


