        subroutine idd_poweroftwo(m,l,n)
c
c       computes l = floor(log_2(m)) and n = 2**l.
c
c       input:
c       m -- integer whose log_2 is to be taken
c
c       output:
c       l -- floor(log_2(m))
c       n -- 2**l
c
        implicit none
        integer l,m,n
c
c
        l = 0
        n = 1
c
 1000   continue
          l = l+1
          n = n*2
        if(n .le. m) goto 1000
c
        l = l-1
        n = n/2
c
c
        return
        end
