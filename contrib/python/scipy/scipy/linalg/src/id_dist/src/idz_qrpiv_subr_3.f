        subroutine idz_qmatmat(ifadjoint,m,n,a,krank,l,b,work)
c
c       applies to a bunch of vectors collected together as a matrix
c       the Q matrix (or its adjoint) which the routine idzp_qrpiv 
c       or idzr_qrpiv has stored in a triangle of the matrix
c       it produces (stored, incidentally, as data
c       for applying a bunch of Householder reflections).
c       Use the routine idz_qmatvec to apply the Q matrix
c       (or its adjoint)
c       to a single vector, if you'd rather not provide a work array.
c
c       input:
c       ifadjoint -- set to 0 for applying Q;
c                    set to 1 for applying the adjoint of Q
c       m -- first dimension of both a and b
c       n -- second dimension of a
c       a -- data describing the qr decomposition of a matrix,
c            as produced by idzp_qrpiv or idzr_qrpiv
c       krank -- numerical rank
c       l -- second dimension of b
c       b -- matrix to which Q (or its adjoint) is to be applied
c
c       output:
c       b -- matrix to which Q (or its adjoint) has been applied
c
c       work:
c       work -- must be at least krank real*8 elements long
c
        implicit none
        save
        integer l,m,n,krank,j,k,ifrescal,mm,ifadjoint
        real*8 work(krank)
        complex*16 a(m,n),b(m,l)
c
c
        if(ifadjoint .eq. 0) then
c
c
c         Handle the first iteration, j = 1,
c         calculating all scals (ifrescal = 1).
c
          ifrescal = 1
c
          j = 1
c
          do k = krank,1,-1
            if(k .lt. m) then
              mm = m-k+1
              call idz_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
     1                          work(k),b(k,j))
            endif
          enddo ! k
c
c
          if(l .gt. 1) then
c
c           Handle the other iterations, j > 1,
c           using the scals just computed (ifrescal = 0).
c
            ifrescal = 0
c
            do j = 2,l
c
              do k = krank,1,-1
                if(k .lt. m) then
                  mm = m-k+1
                  call idz_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
     1                              work(k),b(k,j))
                endif
              enddo ! k
c
            enddo ! j
c
          endif ! j .gt. 1
c
c
        endif ! ifadjoint .eq. 0
c
c
        if(ifadjoint .eq. 1) then
c
c
c         Handle the first iteration, j = 1,
c         calculating all scals (ifrescal = 1).
c
          ifrescal = 1
c
          j = 1
c
          do k = 1,krank
            if(k .lt. m) then
              mm = m-k+1
              call idz_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
     1                          work(k),b(k,j))
            endif
          enddo ! k
c
c
          if(l .gt. 1) then
c
c           Handle the other iterations, j > 1,
c           using the scals just computed (ifrescal = 0).
c
            ifrescal = 0
c
            do j = 2,l
c
              do k = 1,krank
                if(k .lt. m) then
                  mm = m-k+1
                  call idz_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
     1                              work(k),b(k,j))
                endif
              enddo ! k
c
            enddo ! j
c
          endif ! j .gt. 1
c
c
        endif ! ifadjoint .eq. 1
c
c
        return
        end
c
c
c
c
