      SUBROUTINE SLASD7( ICOMPQ, NL, NR, SQRE, K, D, Z, ZW, VF, VFW, VL,
     $                   VLW, ALPHA, BETA, DSIGMA, IDX, IDXP, IDXQ,
     $                   PERM, GIVPTR, GIVCOL, LDGCOL, GIVNUM, LDGNUM,
     $                   C, S, INFO )
*
*  -- LAPACK auxiliary routine (version 3.0) --
*     Univ. of Tennessee, Oak Ridge National Lab, Argonne National Lab,
*     Courant Institute, NAG Ltd., and Rice University
*     June 30, 1999
*
*     .. Scalar Arguments ..
      INTEGER            GIVPTR, ICOMPQ, INFO, K, LDGCOL, LDGNUM, NL,
     $                   NR, SQRE
      REAL               ALPHA, BETA, C, S
*     ..
*     .. Array Arguments ..
      INTEGER            GIVCOL( LDGCOL, * ), IDX( * ), IDXP( * ),
     $                   IDXQ( * ), PERM( * )
      REAL               D( * ), DSIGMA( * ), GIVNUM( LDGNUM, * ),
     $                   VF( * ), VFW( * ), VL( * ), VLW( * ), Z( * ),
     $                   ZW( * )
*     ..
*
*  Purpose
*  =======
*
*  SLASD7 merges the two sets of singular values together into a single
*  sorted set. Then it tries to deflate the size of the problem. There
*  are two ways in which deflation can occur:  when two or more singular
*  values are close together or if there is a tiny entry in the Z
*  vector. For each such occurrence the order of the related
*  secular equation problem is reduced by one.
*
*  SLASD7 is called from SLASD6.
*
*  Arguments
*  =========
*
*  ICOMPQ  (input) INTEGER
*          Specifies whether singular vectors are to be computed
*          in compact form, as follows:
*          = 0: Compute singular values only.
*          = 1: Compute singular vectors of upper
*               bidiagonal matrix in compact form.
*
*  NL     (input) INTEGER
*         The row dimension of the upper block. NL >= 1.
*
*  NR     (input) INTEGER
*         The row dimension of the lower block. NR >= 1.
*
*  SQRE   (input) INTEGER
*         = 0: the lower block is an NR-by-NR square matrix.
*         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
*
*         The bidiagonal matrix has
*         N = NL + NR + 1 rows and
*         M = N + SQRE >= N columns.
*
*  K      (output) INTEGER
*         Contains the dimension of the non-deflated matrix, this is
*         the order of the related secular equation. 1 <= K <=N.
*
*  D      (input/output) REAL array, dimension ( N )
*         On entry D contains the singular values of the two submatrices
*         to be combined. On exit D contains the trailing (N-K) updated
*         singular values (those which were deflated) sorted into
*         increasing order.
*
*  Z      (output) REAL array, dimension ( M )
*         On exit Z contains the updating row vector in the secular
*         equation.
*
*  ZW     (workspace) REAL array, dimension ( M )
*         Workspace for Z.
*
*  VF     (input/output) REAL array, dimension ( M )
*         On entry, VF(1:NL+1) contains the first components of all
*         right singular vectors of the upper block; and VF(NL+2:M)
*         contains the first components of all right singular vectors
*         of the lower block. On exit, VF contains the first components
*         of all right singular vectors of the bidiagonal matrix.
*
*  VFW    (workspace) REAL array, dimension ( M )
*         Workspace for VF.
*
*  VL     (input/output) REAL array, dimension ( M )
*         On entry, VL(1:NL+1) contains the  last components of all
*         right singular vectors of the upper block; and VL(NL+2:M)
*         contains the last components of all right singular vectors
*         of the lower block. On exit, VL contains the last components
*         of all right singular vectors of the bidiagonal matrix.
*
*  VLW    (workspace) REAL array, dimension ( M )
*         Workspace for VL.
*
*  ALPHA  (input) REAL
*         Contains the diagonal element associated with the added row.
*
*  BETA   (input) REAL
*         Contains the off-diagonal element associated with the added
*         row.
*
*  DSIGMA (output) REAL array, dimension ( N )
*         Contains a copy of the diagonal elements (K-1 singular values
*         and one zero) in the secular equation.
*
*  IDX    (workspace) INTEGER array, dimension ( N )
*         This will contain the permutation used to sort the contents of
*         D into ascending order.
*
*  IDXP   (workspace) INTEGER array, dimension ( N )
*         This will contain the permutation used to place deflated
*         values of D at the end of the array. On output IDXP(2:K)
*         points to the nondeflated D-values and IDXP(K+1:N)
*         points to the deflated singular values.
*
*  IDXQ   (input) INTEGER array, dimension ( N )
*         This contains the permutation which separately sorts the two
*         sub-problems in D into ascending order.  Note that entries in
*         the first half of this permutation must first be moved one
*         position backward; and entries in the second half
*         must first have NL+1 added to their values.
*
*  PERM   (output) INTEGER array, dimension ( N )
*         The permutations (from deflation and sorting) to be applied
*         to each singular block. Not referenced if ICOMPQ = 0.
*
*  GIVPTR (output) INTEGER
*         The number of Givens rotations which took place in this
*         subproblem. Not referenced if ICOMPQ = 0.
*
*  GIVCOL (output) INTEGER array, dimension ( LDGCOL, 2 )
*         Each pair of numbers indicates a pair of columns to take place
*         in a Givens rotation. Not referenced if ICOMPQ = 0.
*
*  LDGCOL (input) INTEGER
*         The leading dimension of GIVCOL, must be at least N.
*
*  GIVNUM (output) REAL array, dimension ( LDGNUM, 2 )
*         Each number indicates the C or S value to be used in the
*         corresponding Givens rotation. Not referenced if ICOMPQ = 0.
*
*  LDGNUM (input) INTEGER
*         The leading dimension of GIVNUM, must be at least N.
*
*  C      (output) REAL
*         C contains garbage if SQRE =0 and the C-value of a Givens
*         rotation related to the right null space if SQRE = 1.
*
*  S      (output) REAL
*         S contains garbage if SQRE =0 and the S-value of a Givens
*         rotation related to the right null space if SQRE = 1.
*
*  INFO   (output) INTEGER
*         = 0:  successful exit.
*         < 0:  if INFO = -i, the i-th argument had an illegal value.
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
      REAL               ZERO, ONE, TWO, EIGHT
      PARAMETER          ( ZERO = 0.0E+0, ONE = 1.0E+0, TWO = 2.0E+0,
     $                   EIGHT = 8.0E+0 )
*     ..
*     .. Local Scalars ..
*
      INTEGER            I, IDXI, IDXJ, IDXJP, J, JP, JPREV, K2, M, N,
     $                   NLP1, NLP2
      REAL               EPS, HLFTOL, TAU, TOL, Z1
*     ..
*     .. External Subroutines ..
      EXTERNAL           SCOPY, SLAMRG, SROT, XERBLA
*     ..
*     .. External Functions ..
      REAL               SLAMCH, SLAPY2
      EXTERNAL           SLAMCH, SLAPY2
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      INFO = 0
      N = NL + NR + 1
      M = N + SQRE
*
      IF( ( ICOMPQ.LT.0 ) .OR. ( ICOMPQ.GT.1 ) ) THEN
         INFO = -1
      ELSE IF( NL.LT.1 ) THEN
         INFO = -2
      ELSE IF( NR.LT.1 ) THEN
         INFO = -3
      ELSE IF( ( SQRE.LT.0 ) .OR. ( SQRE.GT.1 ) ) THEN
         INFO = -4
      ELSE IF( LDGCOL.LT.N ) THEN
         INFO = -22
      ELSE IF( LDGNUM.LT.N ) THEN
         INFO = -24
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'SLASD7', -INFO )
         RETURN
      END IF
*
      NLP1 = NL + 1
      NLP2 = NL + 2
      IF( ICOMPQ.EQ.1 ) THEN
         GIVPTR = 0
      END IF
*
*     Generate the first part of the vector Z and move the singular
*     values in the first part of D one position backward.
*
      Z1 = ALPHA*VL( NLP1 )
      VL( NLP1 ) = ZERO
      TAU = VF( NLP1 )
      DO 10 I = NL, 1, -1
         Z( I+1 ) = ALPHA*VL( I )
         VL( I ) = ZERO
         VF( I+1 ) = VF( I )
         D( I+1 ) = D( I )
         IDXQ( I+1 ) = IDXQ( I ) + 1
   10 CONTINUE
      VF( 1 ) = TAU
*
*     Generate the second part of the vector Z.
*
      DO 20 I = NLP2, M
         Z( I ) = BETA*VF( I )
         VF( I ) = ZERO
   20 CONTINUE
*
*     Sort the singular values into increasing order
*
      DO 30 I = NLP2, N
         IDXQ( I ) = IDXQ( I ) + NLP1
   30 CONTINUE
*
*     DSIGMA, IDXC, IDXC, and ZW are used as storage space.
*
      DO 40 I = 2, N
         DSIGMA( I ) = D( IDXQ( I ) )
         ZW( I ) = Z( IDXQ( I ) )
         VFW( I ) = VF( IDXQ( I ) )
         VLW( I ) = VL( IDXQ( I ) )
   40 CONTINUE
*
      CALL SLAMRG( NL, NR, DSIGMA( 2 ), 1, 1, IDX( 2 ) )
*
      DO 50 I = 2, N
         IDXI = 1 + IDX( I )
         D( I ) = DSIGMA( IDXI )
         Z( I ) = ZW( IDXI )
         VF( I ) = VFW( IDXI )
         VL( I ) = VLW( IDXI )
   50 CONTINUE
*
*     Calculate the allowable deflation tolerence
*
      EPS = SLAMCH( 'Epsilon' )
      TOL = MAX( ABS( ALPHA ), ABS( BETA ) )
      TOL = EIGHT*EIGHT*EPS*MAX( ABS( D( N ) ), TOL )
*
*     There are 2 kinds of deflation -- first a value in the z-vector
*     is small, second two (or more) singular values are very close
*     together (their difference is small).
*
*     If the value in the z-vector is small, we simply permute the
*     array so that the corresponding singular value is moved to the
*     end.
*
*     If two values in the D-vector are close, we perform a two-sided
*     rotation designed to make one of the corresponding z-vector
*     entries zero, and then permute the array so that the deflated
*     singular value is moved to the end.
*
*     If there are multiple singular values then the problem deflates.
*     Here the number of equal singular values are found.  As each equal
*     singular value is found, an elementary reflector is computed to
*     rotate the corresponding singular subspace so that the
*     corresponding components of Z are zero in this new basis.
*
      K = 1
      K2 = N + 1
      DO 60 J = 2, N
         IF( ABS( Z( J ) ).LE.TOL ) THEN
*
*           Deflate due to small z component.
*
            K2 = K2 - 1
            IDXP( K2 ) = J
            IF( J.EQ.N )
     $         GO TO 100
         ELSE
            JPREV = J
            GO TO 70
         END IF
   60 CONTINUE
   70 CONTINUE
      J = JPREV
   80 CONTINUE
      J = J + 1
      IF( J.GT.N )
     $   GO TO 90
      IF( ABS( Z( J ) ).LE.TOL ) THEN
*
*        Deflate due to small z component.
*
         K2 = K2 - 1
         IDXP( K2 ) = J
      ELSE
*
*        Check if singular values are close enough to allow deflation.
*
         IF( ABS( D( J )-D( JPREV ) ).LE.TOL ) THEN
*
*           Deflation is possible.
*
            S = Z( JPREV )
            C = Z( J )
*
*           Find sqrt(a**2+b**2) without overflow or
*           destructive underflow.
*
            TAU = SLAPY2( C, S )
            Z( J ) = TAU
            Z( JPREV ) = ZERO
            C = C / TAU
            S = -S / TAU
*
*           Record the appropriate Givens rotation
*
            IF( ICOMPQ.EQ.1 ) THEN
               GIVPTR = GIVPTR + 1
               IDXJP = IDXQ( IDX( JPREV )+1 )
               IDXJ = IDXQ( IDX( J )+1 )
               IF( IDXJP.LE.NLP1 ) THEN
                  IDXJP = IDXJP - 1
               END IF
               IF( IDXJ.LE.NLP1 ) THEN
                  IDXJ = IDXJ - 1
               END IF
               GIVCOL( GIVPTR, 2 ) = IDXJP
               GIVCOL( GIVPTR, 1 ) = IDXJ
               GIVNUM( GIVPTR, 2 ) = C
               GIVNUM( GIVPTR, 1 ) = S
            END IF
            CALL SROT( 1, VF( JPREV ), 1, VF( J ), 1, C, S )
            CALL SROT( 1, VL( JPREV ), 1, VL( J ), 1, C, S )
            K2 = K2 - 1
            IDXP( K2 ) = JPREV
            JPREV = J
         ELSE
            K = K + 1
            ZW( K ) = Z( JPREV )
            DSIGMA( K ) = D( JPREV )
            IDXP( K ) = JPREV
            JPREV = J
         END IF
      END IF
      GO TO 80
   90 CONTINUE
*
*     Record the last singular value.
*
      K = K + 1
      ZW( K ) = Z( JPREV )
      DSIGMA( K ) = D( JPREV )
      IDXP( K ) = JPREV
*
  100 CONTINUE
*
*     Sort the singular values into DSIGMA. The singular values which
*     were not deflated go into the first K slots of DSIGMA, except
*     that DSIGMA(1) is treated separately.
*
      DO 110 J = 2, N
         JP = IDXP( J )
         DSIGMA( J ) = D( JP )
         VFW( J ) = VF( JP )
         VLW( J ) = VL( JP )
  110 CONTINUE
      IF( ICOMPQ.EQ.1 ) THEN
         DO 120 J = 2, N
            JP = IDXP( J )
            PERM( J ) = IDXQ( IDX( JP )+1 )
            IF( PERM( J ).LE.NLP1 ) THEN
               PERM( J ) = PERM( J ) - 1
            END IF
  120    CONTINUE
      END IF
*
*     The deflated singular values go back into the last N - K slots of
*     D.
*
      CALL SCOPY( N-K, DSIGMA( K+1 ), 1, D( K+1 ), 1 )
*
*     Determine DSIGMA(1), DSIGMA(2), Z(1), VF(1), VL(1), VF(M), and
*     VL(M).
*
      DSIGMA( 1 ) = ZERO
      HLFTOL = TOL / TWO
      IF( ABS( DSIGMA( 2 ) ).LE.HLFTOL )
     $   DSIGMA( 2 ) = HLFTOL
      IF( M.GT.N ) THEN
         Z( 1 ) = SLAPY2( Z1, Z( M ) )
         IF( Z( 1 ).LE.TOL ) THEN
            C = ONE
            S = ZERO
            Z( 1 ) = TOL
         ELSE
            C = Z1 / Z( 1 )
            S = -Z( M ) / Z( 1 )
         END IF
         CALL SROT( 1, VF( M ), 1, VF( 1 ), 1, C, S )
         CALL SROT( 1, VL( M ), 1, VL( 1 ), 1, C, S )
      ELSE
         IF( ABS( Z1 ).LE.TOL ) THEN
            Z( 1 ) = TOL
         ELSE
            Z( 1 ) = Z1
         END IF
      END IF
*
*     Restore Z, VF, and VL.
*
      CALL SCOPY( K-1, ZW( 2 ), 1, Z( 2 ), 1 )
      CALL SCOPY( N-1, VFW( 2 ), 1, VF( 2 ), 1 )
      CALL SCOPY( N-1, VLW( 2 ), 1, VL( 2 ), 1 )
*
      RETURN
*
*     End of SLASD7
*
      END
