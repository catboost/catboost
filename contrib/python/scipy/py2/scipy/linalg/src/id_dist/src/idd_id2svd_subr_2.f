        subroutine idd_mattrans(m,n,a,at)
c
c       transposes a to obtain at.
c
c       input:
c       m -- first dimension of a, and second dimension of at
c       n -- second dimension of a, and first dimension of at
c       a -- matrix to be transposed
c
c       output:
c       at -- transpose of a
c
        implicit none
        integer m,n,j,k
        real*8 a(m,n),at(n,m)
c
c
        do k = 1,n
          do j = 1,m
            at(k,j) = a(j,k)
          enddo ! j
        enddo ! k
c
c
        return
        end
c
c
c
c
