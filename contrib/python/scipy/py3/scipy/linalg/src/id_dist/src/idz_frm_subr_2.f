        subroutine idz_subselect(n,ind,m,x,y)
c
c       copies into y the entries of x indicated by ind.
c
c       input:
c       n -- number of entries of x to copy into y
c       ind -- indices of the entries in x to copy into y
c       m -- length of x
c       x -- vector whose entries are to be copied
c
c       output:
c       y -- collection of entries of x specified by ind
c
        implicit none
        integer n,ind(n),m,k
        complex*16 x(m),y(n)
c
c
        do k = 1,n
          y(k) = x(ind(k))
        enddo ! k
c
c
        return
        end
c
c
c
c
