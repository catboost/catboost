        subroutine idz_permuter(krank,ind,m,n,a)
c
c       permutes the columns of a according to ind obtained
c       from routine idzr_qrpiv or idzp_qrpiv, assuming that
c       a = q r from idzr_qrpiv or idzp_qrpiv.
c
c       input:
c       krank -- rank specified to routine idzr_qrpiv
c                or obtained from routine idzp_qrpiv
c       ind -- indexing array obtained from routine idzr_qrpiv
c              or idzp_qrpiv
c       m -- first dimension of a
c       n -- second dimension of a
c       a -- matrix to be rearranged
c
c       output:
c       a -- rearranged matrix
c
        implicit none
        integer k,krank,m,n,j,ind(krank)
        complex*16 cswap,a(m,n)
c
c
        do k = krank,1,-1
          do j = 1,m
c
            cswap = a(j,k)
            a(j,k) = a(j,ind(k))
            a(j,ind(k)) = cswap
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
