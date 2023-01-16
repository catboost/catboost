        subroutine idz_sfft(l,ind,n,wsave,v)
c
c       computes a subset of the entries of the DFT of v,
c       composed with permutation matrices both on input and on output,
c       via a two-stage procedure (routine zfftf2 is supposed
c       to calculate the full vector from which idz_sfft returns
c       a subset of the entries, when zfftf2 has the same parameter
c       nblock as in the present routine).
c
c       input:
c       l -- number of entries in the output to compute
c       ind -- indices of the entries of the output to compute
c       n -- length of v
c       v -- vector to be transformed
c       wsave -- processing array initialized by routine idz_sffti
c
c       output:
c       v -- entries indexed by ind are given their appropriate
c            transformed values
c
c       _N.B._: The user has to boost the memory allocations
c               for wsave (and change iii accordingly) if s/he wishes
c               to use strange sizes of n; it's best to stick to powers
c               of 2.
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
        integer n,m,l,k,j,ind(l),i,idivm,nblock,ii,iii
        real*8 r1,twopi
        complex*16 v(n),wsave(2*l+15+3*n),ci,sum
c
        ci = (0,1)
        r1 = 1
        twopi = 2*4*atan(r1)
c
c
c       Determine the block lengths for the FFTs.
c
        call idz_ldiv(l,n,nblock)
c
c
        m = n/nblock
c
c
c       FFT each block of length nblock of v.
c
        do k = 1,m
          call zfftf(nblock,v(nblock*(k-1)+1),wsave)
        enddo ! k
c
c
c       Transpose v to obtain wsave(2*l+15+2*n+1 : 2*l+15+3*n).
c
        iii = 2*l+15+2*n
c
        do k = 1,m
          do j = 1,nblock
            wsave(iii+m*(j-1)+k) = v(nblock*(k-1)+j)
          enddo ! j
        enddo ! k
c
c
c       Directly calculate the desired entries of v.
c
        ii = 2*l+15
        iii = 2*l+15+2*n
c
        do j = 1,l
c
          i = ind(j)
c
          idivm = (i-1)/m
c
          sum = 0
c
          do k = 1,m
            sum = sum + wsave(ii+m*(j-1)+k) * wsave(iii+m*idivm+k)
          enddo ! k
c
          v(i) = sum
c
        enddo ! j
c
c
        return
        end
