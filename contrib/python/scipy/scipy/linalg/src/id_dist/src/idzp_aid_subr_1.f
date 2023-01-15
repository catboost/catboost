        subroutine idzp_aid0(eps,m,n,a,krank,list,proj,rnorms)
c
c       uses routine idzp_id to ID a without modifying its entries
c       (in contrast to the usual behavior of idzp_id).
c
c       input:
c       eps -- precision of the decomposition to be constructed
c       m -- first dimension of a
c       n -- second dimension of a
c
c       output:
c       krank -- numerical rank of the ID
c       list -- indices of the columns in the ID
c       proj -- matrix of coefficients needed to interpolate
c               from the selected columns to the other columns in a;
c               proj doubles as a work array in the present routine, so
c               must be at least m*n complex*16 elements long
c
c       work:
c       rnorms -- must be at least n real*8 elements long
c
c       _N.B._: proj must be at least m*n complex*16 elements long
c
        implicit none
        integer m,n,krank,list(n),j,k
        real*8 eps,rnorms(n)
        complex*16 a(m,n),proj(m,n)
c
c
c       Copy a into proj.
c
        do k = 1,n
          do j = 1,m
            proj(j,k) = a(j,k)
          enddo ! j
        enddo ! k
c
c
c       ID proj.
c
        call idzp_id(eps,m,n,proj,krank,list,rnorms)
c
c
        return
        end
c
c
c
c
