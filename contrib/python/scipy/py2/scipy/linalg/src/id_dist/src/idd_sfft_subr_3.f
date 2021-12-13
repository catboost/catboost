        subroutine idd_sffti2(l,ind,n,wsave)
c
c       routine idd_sffti serves as a wrapper around
c       the present routine; please see routine idd_sffti
c       for documentation.
c
        implicit none
        integer l,ind(l),n,nblock,ii,m,idivm,imodm,i,j,k
        real*8 r1,twopi,fact
        complex*16 wsave(2*l+15+4*n),ci,twopii
c
        ci = (0,1)
        r1 = 1
        twopi = 2*4*atan(r1)
        twopii = twopi*ci
c
c
c       Determine the block lengths for the FFTs.
c
        call idd_ldiv(l,n,nblock)
        m = n/nblock
c
c
c       Initialize wsave for using routine dfftf.
c
        call dffti(nblock,wsave)
c
c
c       Calculate the coefficients in the linear combinations
c       needed for the direct portion of the calculation.
c
        fact = 1/sqrt(r1*n)
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
            do k = 1,m
              wsave(ii+m*(j-1)+k) = exp(-twopii*(k-1)*imodm/(r1*m))
     1         * exp(-twopii*(k-1)*(idivm+1)/(r1*n)) * fact
            enddo ! k
c
          endif ! i .le. n/2-m/2
c
c
          if(i .gt. n/2-m/2) then
c
            idivm = i/(m/2)
            imodm = i-(m/2)*idivm
c
            do k = 1,m
              wsave(ii+m*(j-1)+k) = exp(-twopii*(k-1)*imodm/(r1*m))
     1                            * fact
            enddo ! k
c
          endif ! i .gt. n/2-m/2
c
c
        enddo ! j
c
c
        return
        end
c
c
c
c
