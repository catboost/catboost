      SUBROUTINE DFFTI (N,WSAVE)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       WSAVE(*)
      IF (N .EQ. 1) RETURN
      CALL DFFTI1 (N,WSAVE(N+1),WSAVE(2*N+1))
      RETURN
      END
      SUBROUTINE DSINQB (N,X,WSAVE)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       X(*)       ,WSAVE(*)
      IF (N .GT. 1) GO TO 101
      X(1) = 4.0D0*X(1)
      RETURN
  101 NS2 = N/2
      DO 102 K=2,N,2
         X(K) = -X(K)
  102 CONTINUE
      CALL DCOSQB (N,X,WSAVE)
      DO 103 K=1,NS2
         KC = N-K
         XHOLD = X(K)
         X(K) = X(KC+1)
         X(KC+1) = XHOLD
  103 CONTINUE
      RETURN
      END
      SUBROUTINE DSINQF (N,X,WSAVE)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       X(*)       ,WSAVE(*)
      IF (N .EQ. 1) RETURN
      NS2 = N/2
      DO 101 K=1,NS2
         KC = N-K
         XHOLD = X(K)
         X(K) = X(KC+1)
         X(KC+1) = XHOLD
  101 CONTINUE
      CALL DCOSQF (N,X,WSAVE)
      DO 102 K=2,N,2
         X(K) = -X(K)
  102 CONTINUE
      RETURN
      END
      SUBROUTINE DSINQI (N,WSAVE)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       WSAVE(*)
      CALL DCOSQI (N,WSAVE)
      RETURN
      END

