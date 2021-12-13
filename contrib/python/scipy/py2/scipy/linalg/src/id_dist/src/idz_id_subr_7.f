        subroutine idz_copycols(m,n,a,krank,list,col)
c
c       collects together the columns of the matrix a indexed by list
c       into the matrix col.
c
c       input:
c       m -- first dimension of a
c       n -- second dimension of a
c       a -- matrix whose columns are to be extracted
c       krank -- number of columns to be extracted
c       list -- indices of the columns to be extracted
c
c       output:
c       col -- columns of a indexed by list
c
        implicit none
        integer m,n,krank,list(krank),j,k
        complex*16 a(m,n),col(m,krank)
c
c
        do k = 1,krank
          do j = 1,m
c
            col(j,k) = a(j,list(k))
c
          enddo ! j
        enddo ! k
c
c
        return
        end
