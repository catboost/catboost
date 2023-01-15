        subroutine idz_sffti(l,ind,n,wsave)
c
c       initializes wsave for use with routine idz_sfft.
c
c       input:
c       l -- number of entries in the output of idz_sfft to compute
c       ind -- indices of the entries in the output of idz_sfft
c              to compute
c       n -- length of the vector to be transformed
c
c       output:
c       wsave -- array needed by routine idz_sfft for processing
c
        implicit none
        integer l,ind(l),n,nblock,ii,m,idivm,imodm,i,j,k
        real*8 r1,twopi,fact
        complex*16 wsave(2*l+15+3*n),ci,twopii
c
        ci = (0,1)
        r1 = 1
        twopi = 2*4*atan(r1)
        twopii = twopi*ci
c
c
c       Determine the block lengths for the FFTs.
c
        call idz_ldiv(l,n,nblock)
        m = n/nblock
c
c
c       Initialize wsave for use with routine zfftf.
c
        call zffti(nblock,wsave)
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
          i = ind(j)
c
          idivm = (i-1)/m
          imodm = (i-1)-m*idivm
c
          do k = 1,m
            wsave(ii+m*(j-1)+k) = exp(-twopii*imodm*(k-1)/(r1*m))
     1       * exp(-twopii*(k-1)*idivm/(r1*n)) * fact
          enddo ! k
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
