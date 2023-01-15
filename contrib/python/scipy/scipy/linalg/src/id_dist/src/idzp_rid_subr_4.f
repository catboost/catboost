        subroutine idz_adjointer(m,n,a,aa)
c
c       forms the adjoint aa of a.
c
c       input:
c       m -- first dimension of a, and second dimension of aa
c       n -- second dimension of a, and first dimension of aa
c       a -- matrix whose adjoint is to be taken
c
c       output:
c       aa -- adjoint of a
c
        implicit none
        integer m,n,j,k
        complex*16 a(m,n),aa(n,m)
c
c
        do k = 1,n
          do j = 1,m
c
            aa(k,j) = conjg(a(j,k))
c
          enddo ! j
        enddo ! k
c
c
        return
        end
