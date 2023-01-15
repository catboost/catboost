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
