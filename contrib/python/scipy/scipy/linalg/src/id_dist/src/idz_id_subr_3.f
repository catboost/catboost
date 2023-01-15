        subroutine idz_lssolve(m,n,a,krank)
c
c       backsolves for proj satisfying R_11 proj ~ R_12,
c       where R_11 = a(1:krank,1:krank)
c       and R_12 = a(1:krank,krank+1:n).
c       This routine overwrites the beginning of a with proj.
c
c       input:
c       m -- first dimension of a
c       n -- second dimension of a; also,
c            n-krank is the second dimension of proj
c       a -- trapezoidal input matrix
c       krank -- first dimension of proj; also,
c                n-krank is the second dimension of proj
c
c       output:
c       a -- the first krank*(n-krank) elements of a constitute
c            the krank x (n-krank) matrix proj
c
        implicit none
        integer m,n,krank,j,k,l
        real*8 rnumer,rdenom
        complex*16 a(m,n),sum
c
c
c       Overwrite a(1:krank,krank+1:n) with proj.
c
        do k = 1,n-krank
          do j = krank,1,-1
c
            sum = 0
c
            do l = j+1,krank
              sum = sum+a(j,l)*a(l,krank+k)
            enddo ! l
c
            a(j,krank+k) = a(j,krank+k)-sum
c
c           Make sure that the entry in proj won't be too big;
c           set the entry to 0 when roundoff would make it too big
c           (in which case a(j,j) is so small that the contribution
c           from this entry in proj to the overall matrix approximation
c           is supposed to be negligible).
c
            rnumer = a(j,krank+k)*conjg(a(j,krank+k))
            rdenom = a(j,j)*conjg(a(j,j))
c
            if(rnumer .lt. 2**30*rdenom) then
              a(j,krank+k) = a(j,krank+k)/a(j,j)
            else
              a(j,krank+k) = 0
            endif
c
          enddo ! j
        enddo ! k
c
c
c       Move proj from a(1:krank,krank+1:n) to the beginning of a.
c
        call idz_moverup(m,n,krank,a)
c
c
        return
        end
c
c
c
c
