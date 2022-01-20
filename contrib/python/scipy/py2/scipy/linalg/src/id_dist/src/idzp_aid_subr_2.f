        subroutine idzp_aid1(eps,n2,n,kranki,proj,krank,list,rnorms)
c
c       IDs the uppermost kranki x n block of the n2 x n matrix
c       input as proj.
c
c       input:
c       eps -- precision of the decomposition to be constructed
c       n2 -- first dimension of proj as input
c       n -- second dimension of proj as input
c       kranki -- number of rows to extract from proj
c       proj -- matrix containing the kranki x n block to be ID'd
c
c       output:
c       proj -- matrix of coefficients needed to interpolate
c               from the selected columns to the other columns
c               in the original matrix being ID'd
c       krank -- numerical rank of the ID
c       list -- indices of the columns in the ID
c
c       work:
c       rnorms -- must be at least n real*8 elements long
c
        implicit none
        integer n,n2,kranki,krank,list(n),j,k
        real*8 eps,rnorms(n)
        complex*16 proj(n2*n)
c
c
c       Move the uppermost kranki x n block of the n2 x n matrix proj
c       to the beginning of proj.
c
        do k = 1,n
          do j = 1,kranki
            proj(j+kranki*(k-1)) = proj(j+n2*(k-1))
          enddo ! j
        enddo ! k
c
c
c       ID proj.
c
        call idzp_id(eps,kranki,n,proj,krank,list,rnorms)
c
c
        return
        end
c
c
c
c
