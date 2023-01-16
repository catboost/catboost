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
