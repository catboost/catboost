        subroutine id_randperm(n,ind)
c
c       draws a permutation ind uniformly at random from the group
c       of all permutations of n objects.
c
c       input:
c       n -- length of ind
c
c       output:
c       ind -- random permutation of length n
c
        implicit none
        integer n,ind(n),m,j,iswap
        real*8 r
c
c
c       Initialize ind.
c
        do j = 1,n
          ind(j) = j
        enddo ! j
c
c
c       Shuffle ind via the Fisher-Yates (Knuth/Durstenfeld) algorithm.
c
        do m = n,2,-1
c
c         Draw an integer uniformly at random from 1, 2, ..., m.
c
          call id_srand(1,r)
          j = m*r+1
c
c         Uncomment the following line if r could equal 1:
c         if(j .eq. m+1) j = m
c
c         Swap ind(j) and ind(m).
c
          iswap = ind(j)
          ind(j) = ind(m)
          ind(m) = iswap
c
        enddo ! m
c
c
        return
        end
