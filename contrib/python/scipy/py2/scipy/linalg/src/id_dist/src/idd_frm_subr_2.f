        subroutine idd_permute(n,ind,x,y)
c
c       copy the entries of x into y, rearranged according
c       to the permutation specified by ind.
c
c       input:
c       n -- length of ind, x, and y
c       ind -- permutation of n objects
c       x -- vector to be permuted
c
c       output:
c       y -- permutation of x
c
        implicit none
        integer n,ind(n),k
        real*8 x(n),y(n)
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
