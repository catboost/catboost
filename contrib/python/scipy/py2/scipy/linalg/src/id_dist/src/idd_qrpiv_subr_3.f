        subroutine idd_qmatmat(iftranspose,m,n,a,krank,l,b,work)
c
c       applies to a bunch of vectors collected together as a matrix
c       the Q matrix (or its transpose) which the routine iddp_qrpiv or
c       iddr_qrpiv has stored in a triangle of the matrix it produces
c       (stored, incidentally, as data for applying a bunch
c       of Householder reflections).
c       Use the routine qmatvec to apply the Q matrix
c       (or its transpose)
c       to a single vector, if you'd rather not provide a work array.
c
c       input:
c       iftranspose -- set to 0 for applying Q;
c                      set to 1 for applying the transpose of Q
c       m -- first dimension of both a and b
c       n -- second dimension of a
c       a -- data describing the qr decomposition of a matrix,
c            as produced by iddp_qrpiv or iddr_qrpiv
c       krank -- numerical rank
c       l -- second dimension of b
c       b -- matrix to which Q (or its transpose) is to be applied
c
c       output:
c       b -- matrix to which Q (or its transpose) has been applied
c
c       work:
c       work -- must be at least krank real*8 elements long
c
        implicit none
        save
        integer l,m,n,krank,j,k,ifrescal,mm,iftranspose
        real*8 a(m,n),b(m,l),work(krank)
c
c
        if(iftranspose .eq. 0) then
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
              call idd_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
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
                  call idd_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
     1                              work(k),b(k,j))
                endif
              enddo ! k
c
            enddo ! j
c
          endif ! j .gt. 1
c
c
        endif ! iftranspose .eq. 0
c
c
        if(iftranspose .eq. 1) then
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
              call idd_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
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
                  call idd_houseapp(mm,a(k+1,k),b(k,j),ifrescal,
     1                              work(k),b(k,j))
                endif
              enddo ! k
c
            enddo ! j
c
          endif ! j .gt. 1
c
c
        endif ! iftranspose .eq. 1
c
c
        return
        end
c
c
c
c
