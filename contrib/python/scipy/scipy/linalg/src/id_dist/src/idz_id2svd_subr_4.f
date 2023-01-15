        subroutine idz_rearr(krank,ind,m,n,a)
c
c       rearranges a according to ind obtained
c       from routines idzr_qrpiv or idzp_qrpiv,
c       assuming that a = q r, where q and r are from idzr_qrpiv
c       or idzp_qrpiv.
c
c       input:
c       krank -- rank obtained from routine idzp_qrpiv,
c                or provided to routine idzr_qrpiv
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
