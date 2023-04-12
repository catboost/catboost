        subroutine idz_transposer(m,n,a,at)
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
        complex*16 a(m,n),at(n,m)
c
c
        do k = 1,n
          do j = 1,m
c
            at(k,j) = a(j,k)
c
          enddo ! j
        enddo ! k
c
c
        return
        end
