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
        subroutine idd_sffti(l,ind,n,wsave)
c
c       initializes wsave for using routine idd_sfft.
c
c       input:
c       l -- number of pairs of entries in the output of idd_sfft
c            to compute
c       ind -- indices of the pairs of entries in the output
c              of idd_sfft to compute; the indices must be chosen
c              in the range from 1 to n/2
c       n -- length of the vector to be transformed
c
c       output:
c       wsave -- array needed by routine idd_sfft for processing
c                (the present routine does not use the last n elements
c                 of wsave, but routine idd_sfft does)
c
        implicit none
        integer l,ind(l),n
        complex*16 wsave(2*l+15+4*n)
c
c
        if(l .eq. 1) call idd_sffti1(ind,n,wsave)
        if(l .gt. 1) call idd_sffti2(l,ind,n,wsave)
c
c
        return
        end
c
c
c
c
