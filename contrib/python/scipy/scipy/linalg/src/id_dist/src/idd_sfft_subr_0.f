        subroutine idd_ldiv(l,n,m)
c
c       finds the greatest integer less than or equal to l
c       that divides n.
c
c       input:
c       l -- integer at least as great as m
c       n -- integer divisible by m
c
c       output:
c       m -- greatest integer less than or equal to l that divides n
c
        implicit none
        integer n,l,m
c
c
        m = l
c
 1000   continue
        if(m*(n/m) .eq. n) goto 2000
c
          m = m-1
          goto 1000
c
 2000   continue
c
c
        return
        end
c
c
c
c
