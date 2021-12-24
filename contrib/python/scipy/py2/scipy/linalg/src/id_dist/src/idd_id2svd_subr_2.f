        subroutine idd_mattrans(m,n,a,at)
c
c       transposes a to obtain at.
c
c       input:
c       m -- first dimension of a, and second dimension of at
c       n -- second dimension of a, and first dimension of at
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
c
c
c
c
        subroutine idd_matmultt(l,m,a,n,b,c)
c
c       multiplies a and b^T to obtain c.
c
c       input:
c       l -- first dimension of a and c
c       m -- second dimension of a and b
c       a -- leftmost matrix in the product c = a b^T
c       n -- first dimension of b and second dimension of c
c       b -- rightmost matrix in the product c = a b^T
c
c       output:
c       c -- product of a and b^T
c
        implicit none
        integer l,m,n,i,j,k
        real*8 a(l,m),b(n,m),c(l,n),sum
c
c
        do i = 1,l
          do k = 1,n
c
            sum = 0
c
            do j = 1,m
              sum = sum+a(i,j)*b(k,j)
            enddo ! j
c
            c(i,k) = sum
c
          enddo ! k
        enddo ! i
c
c
        return
        end
c
c
c
c
        subroutine idd_rearr(krank,ind,m,n,a)
c
c       rearranges a according to ind obtained
c       from routines iddr_qrpiv or iddp_qrpiv,
c       assuming that a = q r, where q and r are from iddr_qrpiv
c       or iddp_qrpiv.
c
c       input:
c       krank -- rank obtained from routine iddp_qrpiv,
c                or provided to routine iddr_qrpiv
c       ind -- indexing array obtained from routine iddr_qrpiv
c              or iddp_qrpiv
c       m -- first dimension of a
c       n -- second dimension of a
c       a -- matrix to be rearranged
c
c       output:
c       a -- rearranged matrix
c
        implicit none
        integer k,krank,m,n,j,ind(krank)
        real*8 rswap,a(m,n)
c
c
        do k = krank,1,-1
          do j = 1,m
c
            rswap = a(j,k)
            a(j,k) = a(j,ind(k))
            a(j,ind(k)) = rswap
c
          enddo ! j
        enddo ! k
c
c
        return
        end
c
c
c
c
        subroutine idd_rinqr(m,n,a,krank,r)
c
c       extracts R in the QR decomposition specified by the output a
c       of the routine iddr_qrpiv or iddp_qrpiv.
c
c       input:
c       m -- first dimension of a
c       n -- second dimension of a and r
c       a -- output of routine iddr_qrpiv or iddp_qrpiv
c       krank -- rank output by routine iddp_qrpiv (or specified
c                to routine iddr_qrpiv)
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
