        subroutine idz_crunch(n,l,a)
c
c       removes every other block of n entries from a vector.
c
c       input:
c       n -- length of each block to remove
c       l -- half of the total number of blocks
c       a -- original array
c
c       output:
c       a -- array with every other block of n entries removed
c
        implicit none
        integer j,k,n,l
        complex*16 a(n,2*l)
c
c
        do j = 2,l
          do k = 1,n
c
            a(k,j) = a(k,2*j-1)
c
          enddo ! k
        enddo ! j
c
c
        return
        end
c
c
c
c
