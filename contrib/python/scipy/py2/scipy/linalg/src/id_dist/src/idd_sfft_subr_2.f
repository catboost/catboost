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
        subroutine idd_sfft(l,ind,n,wsave,v)
c
c       computes a subset of the entries of the DFT of v,
c       composed with permutation matrices both on input and on output,
c       via a two-stage procedure (debugging code routine dfftf2 above
c       is supposed to calculate the full vector from which idd_sfft
c       returns a subset of the entries, when dfftf2 has
c       the same parameter nblock as in the present routine).
c
c       input:
c       l -- number of pairs of entries in the output to compute
c       ind -- indices of the pairs of entries in the output
c              to compute; the indices must be chosen
c              in the range from 1 to n/2
c       n -- length of v; n must be a positive integer power of 2
c       v -- vector to be transformed
c       wsave -- processing array initialized by routine idd_sffti
c
c       output:
c       v -- pairs of entries indexed by ind are given
c            their appropriately transformed values
c
c       _N.B._: n must be a positive integer power of 2.
c
c       references:
c       Sorensen and Burrus, "Efficient computation of the DFT with
c            only a subset of input or output points,"
c            IEEE Transactions on Signal Processing, 41 (3): 1184-1200,
c            1993.
c       Woolfe, Liberty, Rokhlin, Tygert, "A fast randomized algorithm
c            for the approximation of matrices," Applied and
c            Computational Harmonic Analysis, 25 (3): 335-366, 2008;
c            Section 3.3.
c
        implicit none
        integer l,ind(l),n
        real*8 v(n)
        complex*16 wsave(2*l+15+4*n)
c
c
        if(l .eq. 1) call idd_sfft1(ind,n,v,wsave)
        if(l .gt. 1) call idd_sfft2(l,ind,n,v,wsave)
c
c
        return
        end
c
c
c
c
