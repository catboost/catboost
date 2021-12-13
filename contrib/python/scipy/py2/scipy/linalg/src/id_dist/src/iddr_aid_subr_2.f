        subroutine iddr_copydarr(n,a,b)
c
c       copies a into b.
c
c       input:
c       n -- length of a and b
c       a -- array to copy into b
c
c       output:
c       b -- copy of a
c
        implicit none
        integer n,k
        real*8 a(n),b(n)
c
c
        do k = 1,n
          b(k) = a(k)
        enddo ! k
c
c
        return
        end
c
c
c
c
