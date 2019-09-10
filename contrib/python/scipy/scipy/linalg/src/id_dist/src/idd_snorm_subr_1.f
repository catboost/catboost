        subroutine idd_enorm(n,v,enorm)
c
c       computes the Euclidean norm of v, the square root
c       of the sum of the squares of the entries of v.
c
c       input:
c       n -- length of v
c       v -- vector whose Euclidean norm is to be calculated
c
c       output:
c       enorm -- Euclidean norm of v
c
        implicit none
        integer n,k
        real*8 enorm,v(n)
c
c
        enorm = 0
c
        do k = 1,n
          enorm = enorm+v(k)**2
        enddo ! k
c
        enorm = sqrt(enorm)
c
c
        return
        end
c
c
c
c
