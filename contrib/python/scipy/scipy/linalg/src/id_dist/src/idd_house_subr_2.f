        subroutine idd_housemat(n,vn,scal,h)
c
c       fills h with the Householder matrix
c       identity_matrix - scal * vn * transpose(vn).
c
c       input:
c       n -- size of vn and h, though the indexing of vn goes
c            from 2 to n
c       vn -- entries 2 to n of the vector vn;
c             vn(1) is assumed to be 1
c       scal -- scalar multiplying vn * transpose(vn)
c
c       output:
c       h -- identity_matrix - scal * vn * transpose(vn)
c
        implicit none
        save
        integer n,j,k
        real*8 vn(2:*),h(n,n),scal,factor1,factor2
c
c
c       Fill h with the identity matrix.
c
        do j = 1,n
          do k = 1,n
c
            if(j .eq. k) h(k,j) = 1
            if(j .ne. k) h(k,j) = 0
c
          enddo ! k
        enddo ! j
c
c
c       Subtract from h the matrix scal*vn*transpose(vn).
c
        do j = 1,n
          do k = 1,n
c
            if(j .eq. 1) factor1 = 1
            if(j .ne. 1) factor1 = vn(j)
c
            if(k .eq. 1) factor2 = 1
            if(k .ne. 1) factor2 = vn(k)
c
            h(k,j) = h(k,j) - scal*factor1*factor2
c
          enddo ! k
        enddo ! j
c
c
        return
        end
