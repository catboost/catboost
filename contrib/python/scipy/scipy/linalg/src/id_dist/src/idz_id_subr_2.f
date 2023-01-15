        subroutine idz_reconid(m,krank,col,n,list,proj,approx)
c
c       reconstructs the matrix that the routine idzp_id
c       or idzr_id has decomposed, using the columns col
c       of the reconstructed matrix whose indices are listed in list,
c       in addition to the interpolation matrix proj.
c
c       input:
c       m -- first dimension of cols and approx
c       krank -- first dimension of cols and proj; also,
c                n-krank is the second dimension of proj
c       col -- columns of the matrix to be reconstructed
c       n -- second dimension of approx; also,
c            n-krank is the second dimension of proj
c       list(k) -- index of col(1:m,k) in the reconstructed matrix
c                  when k <= krank; in general, list describes
c                  the permutation required for reconstruction
c                  via cols and proj
c       proj -- interpolation matrix
c
c       output:
c       approx -- reconstructed matrix
c
        implicit none
        integer m,n,krank,j,k,l,list(n)
        complex*16 col(m,krank),proj(krank,n-krank),approx(m,n)
c
c
        do j = 1,m
          do k = 1,n
c
            approx(j,list(k)) = 0
c
c           Add in the contributions due to the identity matrix.
c
            if(k .le. krank) then
              approx(j,list(k)) = approx(j,list(k)) + col(j,k)
            endif
c
c           Add in the contributions due to proj.
c
            if(k .gt. krank) then
              if(krank .gt. 0) then
c
                do l = 1,krank
                  approx(j,list(k)) = approx(j,list(k))
     1                              + col(j,l)*proj(l,k-krank)
                enddo ! l
c
              endif
            endif
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
