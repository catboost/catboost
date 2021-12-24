        subroutine idd_permuter(krank,ind,m,n,a)
c
c       permutes the columns of a according to ind obtained
c       from routine iddr_qrpiv or iddp_qrpiv, assuming that
c       a = q r from iddr_qrpiv or iddp_qrpiv.
c
c       input:
c       krank -- rank specified to routine iddr_qrpiv
c                or obtained from routine iddp_qrpiv
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
