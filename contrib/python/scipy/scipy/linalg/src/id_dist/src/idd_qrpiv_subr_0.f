        subroutine idd_permmult(m,ind,n,indprod)
c
c       multiplies together the series of permutations in ind.
c
c       input:
c       m -- length of ind
c       ind(k) -- number of the slot with which to swap
c                 the k^th slot
c       n -- length of indprod and indprodinv
c
c       output:
c       indprod -- product of the permutations in ind,
c                  with the permutation swapping 1 and ind(1)
c                  taken leftmost in the product,
c                  that swapping 2 and ind(2) taken next leftmost,
c                  ..., that swapping krank and ind(krank)
c                  taken rightmost; indprod(k) is the number
c                  of the slot with which to swap the k^th slot
c                  in the product permutation
c
        implicit none
        integer m,n,ind(m),indprod(n),k,iswap
c
c
        do k = 1,n
          indprod(k) = k
        enddo ! k 
c
        do k = m,1,-1
c
c         Swap indprod(k) and indprod(ind(k)).
c
          iswap = indprod(k)
          indprod(k) = indprod(ind(k))
          indprod(ind(k)) = iswap
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
