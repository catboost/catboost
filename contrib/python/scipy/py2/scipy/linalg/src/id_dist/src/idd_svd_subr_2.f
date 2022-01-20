        subroutine idd_retriever(m,n,a,krank,r)
c
c       extracts R in the QR decomposition specified by the output a
c       of the routine iddr_qrpiv or iddp_qrpiv
c
c       input:
c       m -- first dimension of a
c       n -- second dimension of a and r
c       a -- output of routine iddr_qrpiv or iddp_qrpiv
c       krank -- rank specified to routine iddr_qrpiv,
c                or output by routine iddp_qrpiv
c
c       output:
c       r -- triangular factor in the QR decomposition specified
c            by the output a of the routine iddr_qrpiv or iddp_qrpiv
c
        implicit none
        integer m,n,j,k,krank
        real*8 a(m,n),r(krank,n)
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
        subroutine idd_transer(m,n,a,at)
c
c       forms the transpose at of a.
c
c       input:
c       m -- first dimension of a and second dimension of at
c       n -- second dimension of a and first dimension of at
c       a -- matrix to be transposed
c
c       output:
c       at -- transpose of a
c
        implicit none
        integer m,n,j,k
        real*8 a(m,n),at(n,m)
c
c
        do k = 1,n
          do j = 1,m
            at(k,j) = a(j,k)
          enddo ! j
        enddo ! k
c
c
        return
        end
