        subroutine idd_reconint(n,list,krank,proj,p)
c
c       constructs p in the ID a = b p,
c       where the columns of b are a subset of the columns of a,
c       and p is the projection coefficient matrix,
c       given list, krank, and proj output
c       by routines iddp_id or iddr_id.
c
c       input:
c       n -- part of the second dimension of proj and p
c       list -- list of columns retained from the original matrix
c               in the ID
c       krank -- rank of the ID
c       proj -- matrix of projection coefficients in the ID
c
c       output:
c       p -- projection matrix in the ID
c
        implicit none
        integer n,krank,list(n),j,k
        real*8 proj(krank,n-krank),p(krank,n)
c
c
        do k = 1,krank
          do j = 1,n
c
            if(j .le. krank) then
              if(j .eq. k) p(k,list(j)) = 1
              if(j .ne. k) p(k,list(j)) = 0
            endif
c
            if(j .gt. krank) then
              p(k,list(j)) = proj(k,j-krank)
            endif
c
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
