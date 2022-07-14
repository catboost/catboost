        subroutine idd_pairsamps(n,l,ind,l2,ind2,marker)
c
c       calculates the indices of the l2 pairs of integers
c       to which the l individual integers from ind belong.
c       The integers in ind may range from 1 to n.
c
c       input:
c       n -- upper bound on the integers in ind
c            (the number 1 must be a lower bound);
c            n must be even
c       l -- length of ind
c       ind -- integers selected from 1 to n
c
c       output:
c       l2 -- length of ind2
c       ind2 -- indices in the range from 1 to n/2 of the pairs
c               of integers to which the entries of ind belong
c
c       work:
c       marker -- must be at least n/2 integer elements long
c
c       _N.B._: n must be even.
c
        implicit none
        integer l,n,ind(l),ind2(l),marker(n/2),l2,k
c
c
c       Unmark all pairs.
c
        do k = 1,n/2
          marker(k) = 0
        enddo ! k
c
c
c       Mark the required pairs.
c
        do k = 1,l
          marker((ind(k)+1)/2) = marker((ind(k)+1)/2)+1
        enddo ! k
c
c
c       Record the required pairs in indpair.
c
        l2 = 0
c
        do k = 1,n/2
c
          if(marker(k) .ne. 0) then
            l2 = l2+1
            ind2(l2) = k
          endif
c
        enddo ! k
c
c
        return
        end
c
c
c
c
