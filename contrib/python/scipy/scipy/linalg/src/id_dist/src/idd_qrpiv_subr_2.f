        subroutine idd_qmatvec(iftranspose,m,n,a,krank,v)
c
c       applies to a single vector the Q matrix (or its transpose)
c       which the routine iddp_qrpiv or iddr_qrpiv has stored
c       in a triangle of the matrix it produces (stored, incidentally,
c       as data for applying a bunch of Householder reflections).
c       Use the routine qmatmat to apply the Q matrix
c       (or its transpose)
c       to a bunch of vectors collected together as a matrix,
c       if you're concerned about efficiency.
c
c       input:
c       iftranspose -- set to 0 for applying Q;
c                      set to 1 for applying the transpose of Q
c       m -- first dimension of a and length of v
c       n -- second dimension of a
c       a -- data describing the qr decomposition of a matrix,
c            as produced by iddp_qrpiv or iddr_qrpiv
c       krank -- numerical rank
c       v -- vector to which Q (or its transpose) is to be applied
c
c       output:
c       v -- vector to which Q (or its transpose) has been applied
c
        implicit none
        save
        integer m,n,krank,k,ifrescal,mm,iftranspose
        real*8 a(m,n),v(m),scal
c
c
        ifrescal = 1
c
c
        if(iftranspose .eq. 0) then
c
          do k = krank,1,-1
            mm = m-k+1
            if(k .lt. m)
     1       call idd_houseapp(mm,a(k+1,k),v(k),ifrescal,scal,v(k))
          enddo ! k
c
        endif
c
c
        if(iftranspose .eq. 1) then
c
          do k = 1,krank
            mm = m-k+1
            if(k .lt. m)
     1       call idd_houseapp(mm,a(k+1,k),v(k),ifrescal,scal,v(k))
          enddo ! k
c
        endif
c
c
        return
        end
c
c
c
c
