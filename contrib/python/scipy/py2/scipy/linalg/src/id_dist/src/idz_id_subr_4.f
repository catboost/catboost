        subroutine idz_moverup(m,n,krank,a)
c
c       moves the krank x (n-krank) matrix in a(1:krank,krank+1:n),
c       where a is initially dimensioned m x n, to the beginning of a.
c       (This is not the most natural way to code the move,
c       but one of my usually well-behaved compilers chokes
c       on more natural ways.)
c
c       input:
c       m -- initial first dimension of a
c       n -- initial second dimension of a
c       krank -- number of rows to move
c       a -- m x n matrix whose krank x (n-krank) block
c            a(1:krank,krank+1:n) is to be moved
c
c       output:
c       a -- array starting with the moved krank x (n-krank) block
c
        implicit none
        integer m,n,krank,j,k
        complex*16 a(m*n)
c
c
        do k = 1,n-krank
          do j = 1,krank
            a(j+krank*(k-1)) = a(j+m*(krank+k-1))
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
