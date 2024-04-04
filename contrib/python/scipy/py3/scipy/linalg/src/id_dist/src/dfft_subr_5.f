      SUBROUTINE ZFFTI (N,WSAVE)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       WSAVE(*)
      IF (N .EQ. 1) RETURN
      IW1 = N+N+1
      IW2 = IW1+N+N
      CALL ZFFTI1 (N,WSAVE(IW1),WSAVE(IW2))
      RETURN
      END

      SUBROUTINE DCOSQB1 (N,X,W,XH)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       X(*)       ,W(*)       ,XH(*)
      NS2 = (N+1)/2
      NP2 = N+2
      DO 101 I=3,N,2
         XIM1 = X(I-1)+X(I)
         X(I) = X(I)-X(I-1)
         X(I-1) = XIM1
  101 CONTINUE
      X(1) = X(1)+X(1)
      MODN = MOD(N,2)
      IF (MODN .EQ. 0) X(N) = X(N)+X(N)
      CALL DFFTB (N,X,XH)
      DO 102 K=2,NS2
         KC = NP2-K
         XH(K) = W(K-1)*X(KC)+W(KC-1)*X(K)
         XH(KC) = W(K-1)*X(K)-W(KC-1)*X(KC)
  102 CONTINUE
      IF (MODN .EQ. 0) X(NS2+1) = W(NS2)*(X(NS2+1)+X(NS2+1))
      DO 103 K=2,NS2
         KC = NP2-K
         X(K) = XH(K)+XH(KC)
         X(KC) = XH(K)-XH(KC)
  103 CONTINUE
      X(1) = X(1)+X(1)
      RETURN
      END

      SUBROUTINE DCOSQF1 (N,X,W,XH)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       X(*)       ,W(*)       ,XH(*)
      NS2 = (N+1)/2
      NP2 = N+2
      DO 101 K=2,NS2
         KC = NP2-K
         XH(K) = X(K)+X(KC)
         XH(KC) = X(K)-X(KC)
  101 CONTINUE
      MODN = MOD(N,2)
      IF (MODN .EQ. 0) XH(NS2+1) = X(NS2+1)+X(NS2+1)
      DO 102 K=2,NS2
         KC = NP2-K
         X(K) = W(K-1)*XH(KC)+W(KC-1)*XH(K)
         X(KC) = W(K-1)*XH(K)-W(KC-1)*XH(KC)
  102 CONTINUE
      IF (MODN .EQ. 0) X(NS2+1) = W(NS2)*XH(NS2+1)
      CALL DFFTF (N,X,XH)
      DO 103 I=3,N,2
         XIM1 = X(I-1)-X(I)
         X(I) = X(I-1)+X(I)
         X(I-1) = XIM1
  103 CONTINUE
      RETURN
      END
      SUBROUTINE DCOSQI (N,WSAVE)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       WSAVE(*)
      DATA PIH /1.5707963267948966192313216916397514D0/
      DT = PIH/DBLE(N)
      FK = 0.0D0
      DO 101 K=1,N
         FK = FK+1.0D0
         WSAVE(K) = DCOS(FK*DT)
  101 CONTINUE
      CALL DFFTI (N,WSAVE(N+1))
      RETURN
      END
      SUBROUTINE DCOST (N,X,WSAVE)
	IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION       X(*)       ,WSAVE(*)
      NM1 = N-1
      NP1 = N+1
      NS2 = N/2
      IF (N-2) 106,101,102
  101 X1H = X(1)+X(2)
      X(2) = X(1)-X(2)
      X(1) = X1H
      RETURN
  102 IF (N .GT. 3) GO TO 103
      X1P3 = X(1)+X(3)
      TX2 = X(2)+X(2)
      X(2) = X(1)-X(3)
      X(1) = X1P3+TX2
      X(3) = X1P3-TX2
      RETURN
  103 C1 = X(1)-X(N)
      X(1) = X(1)+X(N)
      DO 104 K=2,NS2
         KC = NP1-K
         T1 = X(K)+X(KC)
         T2 = X(K)-X(KC)
         C1 = C1+WSAVE(KC)*T2
         T2 = WSAVE(K)*T2
         X(K) = T1-T2
         X(KC) = T1+T2
  104 CONTINUE
      MODN = MOD(N,2)
      IF (MODN .NE. 0) X(NS2+1) = X(NS2+1)+X(NS2+1)
      CALL DFFTF (NM1,X,WSAVE(N+1))
      XIM2 = X(2)
      X(2) = C1
      DO 105 I=4,N,2
         XI = X(I)
         X(I) = X(I-2)-X(I-1)
         X(I-1) = XIM2
         XIM2 = XI
  105 CONTINUE
      IF (MODN .NE. 0) X(N) = XIM2
  106 RETURN
      END

