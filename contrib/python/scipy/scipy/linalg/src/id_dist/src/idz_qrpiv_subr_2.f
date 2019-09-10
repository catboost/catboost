        subroutine idz_qmatvec(ifadjoint,m,n,a,krank,v)
c
c       applies to a single vector the Q matrix (or its adjoint)
c       which the routine idzp_qrpiv or idzr_qrpiv has stored
c       in a triangle of the matrix it produces (stored, incidentally,
c       as data for applying a bunch of Householder reflections).
c       Use the routine idz_qmatmat to apply the Q matrix
c       (or its adjoint)
c       to a bunch of vectors collected together as a matrix,
c       if you're concerned about efficiency.
c
c       input:
c       ifadjoint -- set to 0 for applying Q;
c                    set to 1 for applying the adjoint of Q
c       m -- first dimension of a and length of v
c       n -- second dimension of a
c       a -- data describing the qr decomposition of a matrix,
c            as produced by idzp_qrpiv or idzr_qrpiv
c       krank -- numerical rank
c       v -- vector to which Q (or its adjoint) is to be applied
c
c       output:
c       v -- vector to which Q (or its adjoint) has been applied
c
        implicit none
        save
        integer m,n,krank,k,ifrescal,mm,ifadjoint
        real*8 scal
        complex*16 a(m,n),v(m)
c
c
        ifrescal = 1
c
c
        if(ifadjoint .eq. 0) then
c
          do k = krank,1,-1
            mm = m-k+1
            if(k .lt. m) call idz_houseapp(mm,a(k+1,k),v(k),
     1                                     ifrescal,scal,v(k))
          enddo ! k
c
        endif
c
c
        if(ifadjoint .eq. 1) then
c
          do k = 1,krank
            mm = m-k+1
            if(k .lt. m) call idz_houseapp(mm,a(k+1,k),v(k),
     1                                     ifrescal,scal,v(k))
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
