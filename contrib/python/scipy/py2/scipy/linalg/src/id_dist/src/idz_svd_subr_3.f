        subroutine idz_retriever(m,n,a,krank,r)
c
c       extracts R in the QR decomposition specified by the output a
c       of the routine idzr_qrpiv or idzp_qrpiv
c
c       input:
c       m -- first dimension of a
c       n -- second dimension of a and r
c       a -- output of routine idzr_qrpiv or idzp_qrpiv
c       krank -- rank specified to routine idzr_qrpiv,
c                or output by routine idzp_qrpiv
c
c       output:
c       r -- triangular factor in the QR decomposition specified
c            by the output a of the routine idzr_qrpiv or idzp_qrpiv
c
        implicit none
        integer m,n,j,k,krank
        complex*16 a(m,n),r(krank,n)
c
c
c       Copy a into r and zero out the appropriate
c       Householder vectors that are stored in one triangle of a.
c
        do k = 1,n
          do j = 1,krank
            r(j,k) = a(j,k)
          enddo ! j
        enddo ! k
c
        do k = 1,n
          if(k .lt. krank) then
            do j = k+1,krank
              r(j,k) = 0
            enddo ! j
          endif
        enddo ! k
c
c
        return
        end
c
c
c
c
        subroutine idz_adjer(m,n,a,aa)
c
c       forms the adjoint aa of a.
c
c       input:
c       m -- first dimension of a and second dimension of aa
c       n -- second dimension of a and first dimension of aa
c       a -- matrix whose adjoint is to be taken
c
c       output:
c       aa -- adjoint of a
c
        implicit none
        integer m,n,j,k
        complex*16 a(m,n),aa(n,m)
c
c
        do k = 1,n
          do j = 1,m
            aa(k,j) = conjg(a(j,k))
          enddo ! j
        enddo ! k
c
c
        return
        end
