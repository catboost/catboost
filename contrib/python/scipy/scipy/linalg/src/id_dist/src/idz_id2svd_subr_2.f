        subroutine idz_matadj(m,n,a,aa)
c
c       Takes the adjoint of a to obtain aa.
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
            aa(k,j) = conjg(a(j,k))
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
