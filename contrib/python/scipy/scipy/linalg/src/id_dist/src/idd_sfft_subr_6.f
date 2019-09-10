        subroutine idd_sfft2(l,ind,n,v,wsave)
c
c       routine idd_sfft serves as a wrapper around
c       the present routine; please see routine idd_sfft
c       for documentation.
c
        implicit none
        integer n,m,l,k,j,ind(l),i,idivm,nblock,ii,iii,imodm
        real*8 r1,twopi,v(n),rsum,fact
        complex*16 wsave(2*l+15+4*n),ci,sum
c
        ci = (0,1)
        r1 = 1
        twopi = 2*4*atan(r1)
c
c
c       Determine the block lengths for the FFTs.
c
        call idd_ldiv(l,n,nblock)
c
c
        m = n/nblock
c
c
c       FFT each block of length nblock of v.
c
        do k = 1,m
          call dfftf(nblock,v(nblock*(k-1)+1),wsave)
        enddo ! k
c
c
c       Transpose v to obtain wsave(2*l+15+2*n+1 : 2*l+15+3*n).
c
        iii = 2*l+15+2*n
c
        do k = 1,m
          do j = 1,nblock/2-1
            wsave(iii+m*(j-1)+k) = v(nblock*(k-1)+2*j)
     1                           + ci*v(nblock*(k-1)+2*j+1)
          enddo ! j
        enddo ! k
c
c       Handle the purely real frequency components separately.
c
        do k = 1,m
          wsave(iii+m*(nblock/2-1)+k) = v(nblock*(k-1)+nblock)
          wsave(iii+m*(nblock/2)+k) = v(nblock*(k-1)+1)
        enddo ! k
c
c
c       Directly calculate the desired entries of v.
c
        ii = 2*l+15
c
        do j = 1,l
c
c
          i = ind(j)
c
c
          if(i .le. n/2-m/2) then
c
            idivm = (i-1)/m
            imodm = (i-1)-m*idivm
c
            sum = 0
c
            do k = 1,m
              sum = sum + wsave(iii+m*idivm+k) * wsave(ii+m*(j-1)+k)
            enddo ! k
c
            v(2*i-1) = sum
            v(2*i) = -ci*sum
c
          endif ! i .le. n/2-m/2
c
c
          if(i .gt. n/2-m/2) then
c
            if(i .lt. n/2) then
c
              idivm = i/(m/2)
              imodm = i-(m/2)*idivm
c
              sum = 0
c
              do k = 1,m
                sum = sum + wsave(iii+m*(nblock/2)+k)
     1              * wsave(ii+m*(j-1)+k)
              enddo ! k
c
              v(2*i-1) = sum
              v(2*i) = -ci*sum
c
            endif
c
            if(i .eq. n/2) then
c
              fact = 1/sqrt(r1*n)
c
c
              rsum = 0
c
              do k = 1,m
                rsum = rsum + wsave(iii+m*(nblock/2)+k)
              enddo ! k
c
              v(n-1) = rsum*fact
c
c
              rsum = 0
c
              do k = 1,m/2
                rsum = rsum + wsave(iii+m*(nblock/2)+2*k-1)
                rsum = rsum - wsave(iii+m*(nblock/2)+2*k)
              enddo ! k
c
              v(n) = rsum*fact
c
            endif
c
          endif ! i .gt. n/2-m/2
c
c
        enddo ! j
c
c
        return
        end
