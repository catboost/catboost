      SUBROUTINE DLASD1( NL, NR, SQRE, D, ALPHA, BETA, U, LDU, VT, LDVT,
     $                   IDXQ, IWORK, WORK, INFO )
*
*  -- LAPACK auxiliary routine (version 3.0) --
*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
*     Courant Institute, Argonne National Lab, and Rice University
*     June 30, 1999
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LDU, LDVT, NL, NR, SQRE
      DOUBLE PRECISION   ALPHA, BETA
*     ..
*     .. Array Arguments ..
      INTEGER            IDXQ( * ), IWORK( * )
      DOUBLE PRECISION   D( * ), U( LDU, * ), VT( LDVT, * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DLASD1 computes the SVD of an upper bidiagonal N-by-M matrix B,
*  where N = NL + NR + 1 and M = N + SQRE. DLASD1 is called from DLASD0.
*
*  A related subroutine DLASD7 handles the case in which the singular
*  values (and the singular vectors in factored form) are desired.
*
*  DLASD1 computes the SVD as follows:
*
*                ( D1(in)  0    0     0 )
*    B = U(in) * (   Z1'   a   Z2'    b ) * VT(in)
*                (   0     0   D2(in) 0 )
*
*      = U(out) * ( D(out) 0) * VT(out)
*
*  where Z' = (Z1' a Z2' b) = u' VT', and u is a vector of dimension M
*  with ALPHA and BETA in the NL+1 and NL+2 th entries and zeros
*  elsewhere; and the entry b is empty if SQRE = 0.
*
*  The left singular vectors of the original matrix are stored in U, and
*  the transpose of the right singular vectors are stored in VT, and the
*  singular values are in D.  The algorithm consists of three stages:
*
*     The first stage consists of deflating the size of the problem
*     when there are multiple singular values or when there are zeros in
*     the Z vector.  For each such occurence the dimension of the
*     secular equation problem is reduced by one.  This stage is
*     performed by the routine DLASD2.
*
*     The second stage consists of calculating the updated
*     singular values. This is done by finding the square roots of the
*     roots of the secular equation via the routine DLASD4 (as called
*     by DLASD3). This routine also calculates the singular vectors of
*     the current problem.
*
*     The final stage consists of computing the updated singular vectors
*     directly using the updated singular values.  The singular vectors
*     for the current problem are multiplied with the singular vectors
*     from the overall problem.
*
*  Arguments
*  =========
*
*  NL     (input) INTEGER
*         The row dimension of the upper block.  NL >= 1.
*
*  NR     (input) INTEGER
*         The row dimension of the lower block.  NR >= 1.
*
*  SQRE   (input) INTEGER
*         = 0: the lower block is an NR-by-NR square matrix.
*         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
*
*         The bidiagonal matrix has row dimension N = NL + NR + 1,
*         and column dimension M = N + SQRE.
*
*  D      (input/output) DOUBLE PRECISION array,
*                        dimension (N = NL+NR+1).
*         On entry D(1:NL,1:NL) contains the singular values of the
*         upper block; and D(NL+2:N) contains the singular values of
*         the lower block. On exit D(1:N) contains the singular values
*         of the modified matrix.
*
*  ALPHA  (input) DOUBLE PRECISION
*         Contains the diagonal element associated with the added row.
*
*  BETA   (input) DOUBLE PRECISION
*         Contains the off-diagonal element associated with the added
*         row.
*
*  U      (input/output) DOUBLE PRECISION array, dimension(LDU,N)
*         On entry U(1:NL, 1:NL) contains the left singular vectors of
*         the upper block; U(NL+2:N, NL+2:N) contains the left singular
*         vectors of the lower block. On exit U contains the left
*         singular vectors of the bidiagonal matrix.
*
*  LDU    (input) INTEGER
*         The leading dimension of the array U.  LDU >= max( 1, N ).
*
*  VT     (input/output) DOUBLE PRECISION array, dimension(LDVT,M)
*         where M = N + SQRE.
*         On entry VT(1:NL+1, 1:NL+1)' contains the right singular
*         vectors of the upper block; VT(NL+2:M, NL+2:M)' contains
*         the right singular vectors of the lower block. On exit
*         VT' contains the right singular vectors of the
*         bidiagonal matrix.
*
*  LDVT   (input) INTEGER
*         The leading dimension of the array VT.  LDVT >= max( 1, M ).
*
*  IDXQ  (output) INTEGER array, dimension(N)
*         This contains the permutation which will reintegrate the
*         subproblem just solved back into sorted order, i.e.
*         D( IDXQ( I = 1, N ) ) will be in ascending order.
*
*  IWORK  (workspace) INTEGER array, dimension( 4 * N )
*
*  WORK   (workspace) DOUBLE PRECISION array, dimension( 3*M**2 + 2*M )
*
*  INFO   (output) INTEGER
*          = 0:  successful exit.
*          < 0:  if INFO = -i, the i-th argument had an illegal value.
*          > 0:  if INFO = 1, an singular value did not converge
*
*  Further Details
*  ===============
*
*  Based on contributions by
*     Ming Gu and Huan Ren, Computer Science Division, University of
*     California at Berkeley, USA
*
*  =====================================================================
*
*     .. Parameters ..
*
      DOUBLE PRECISION   ONE, ZERO
      PARAMETER          ( ONE = 1.0D+0, ZERO = 0.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            COLTYP, I, IDX, IDXC, IDXP, IQ, ISIGMA, IU2,
     $                   IVT2, IZ, K, LDQ, LDU2, LDVT2, M, N, N1, N2
      DOUBLE PRECISION   ORGNRM
*     ..
*     .. External Subroutines ..
      EXTERNAL           DLAMRG, DLASCL, DLASD2, DLASD3, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
*
      IF( NL.LT.1 ) THEN
         INFO = -1
      ELSE IF( NR.LT.1 ) THEN
         INFO = -2
      ELSE IF( ( SQRE.LT.0 ) .OR. ( SQRE.GT.1 ) ) THEN
         INFO = -3
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DLASD1', -INFO )
         RETURN
      END IF
*
      N = NL + NR + 1
      M = N + SQRE
*
*     The following values are for bookkeeping purposes only.  They are
*     integer pointers which indicate the portion of the workspace
*     used by a particular array in DLASD2 and DLASD3.
*
      LDU2 = N
      LDVT2 = M
*
      IZ = 1
      ISIGMA = IZ + M
      IU2 = ISIGMA + N
      IVT2 = IU2 + LDU2*N
      IQ = IVT2 + LDVT2*M
*
      IDX = 1
      IDXC = IDX + N
      COLTYP = IDXC + N
      IDXP = COLTYP + N
*
*     Scale.
*
      ORGNRM = MAX( ABS( ALPHA ), ABS( BETA ) )
      D( NL+1 ) = ZERO
      DO 10 I = 1, N
         IF( ABS( D( I ) ).GT.ORGNRM ) THEN
            ORGNRM = ABS( D( I ) )
         END IF
   10 CONTINUE
      CALL DLASCL( 'G', 0, 0, ORGNRM, ONE, N, 1, D, N, INFO )
      ALPHA = ALPHA / ORGNRM
      BETA = BETA / ORGNRM
*
*     Deflate singular values.
*
      CALL DLASD2( NL, NR, SQRE, K, D, WORK( IZ ), ALPHA, BETA, U, LDU,
     $             VT, LDVT, WORK( ISIGMA ), WORK( IU2 ), LDU2,
     $             WORK( IVT2 ), LDVT2, IWORK( IDXP ), IWORK( IDX ),
     $             IWORK( IDXC ), IDXQ, IWORK( COLTYP ), INFO )
*
*     Solve Secular Equation and update singular vectors.
*
      LDQ = K
      CALL DLASD3( NL, NR, SQRE, K, D, WORK( IQ ), LDQ, WORK( ISIGMA ),
     $             U, LDU, WORK( IU2 ), LDU2, VT, LDVT, WORK( IVT2 ),
     $             LDVT2, IWORK( IDXC ), IWORK( COLTYP ), WORK( IZ ),
     $             INFO )
      IF( INFO.NE.0 ) THEN
         RETURN
      END IF
*
*     Unscale.
*
      CALL DLASCL( 'G', 0, 0, ONE, ORGNRM, N, 1, D, N, INFO )
*
*     Prepare the IDXQ sorting permutation.
*
      N1 = K
      N2 = N - K
      CALL DLAMRG( N1, N2, D, 1, -1, IDXQ )
*
      RETURN
*
*     End of DLASD1
*
      END
