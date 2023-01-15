        subroutine idz_getcols(m,n,matvec,p1,p2,p3,p4,krank,list,
     1                         col,x)
c
c       collects together the columns of the matrix a indexed by list
c       into the matrix col, where routine matvec applies a
c       to an arbitrary vector.
c
c       input:
c       m -- first dimension of a
c       n -- second dimension of a
c       matvec -- routine which applies a to an arbitrary vector;
c                 this routine must have a calling sequence of the form
c
c                 matvec(m,x,n,y,p1,p2,p3,p4)
c
c                 where m is the length of x,
c                 x is the vector to which the matrix is to be applied,
c                 n is the length of y,
c                 y is the product of the matrix and x,
c                 and p1, p2, p3, and p4 are user-specified parameters
c       p1 -- parameter to be passed to routine matvec
c       p2 -- parameter to be passed to routine matvec
c       p3 -- parameter to be passed to routine matvec
c       p4 -- parameter to be passed to routine matvec
c       krank -- number of columns to be extracted
c       list -- indices of the columns to be extracted
c
c       output:
c       col -- columns of a indexed by list
c
c       work:
c       x -- must be at least n complex*16 elements long
c
        implicit none
        integer m,n,krank,list(krank),j,k
        complex*16 col(m,krank),x(n),p1,p2,p3,p4
        external matvec
c
c
        do j = 1,krank
c
          do k = 1,n
            x(k) = 0
          enddo ! k
c
          x(list(j)) = 1
c
          call matvec(n,x,m,col(1,j),p1,p2,p3,p4)
c
        enddo ! j
c
c
        return
        end
c
c
c
c
