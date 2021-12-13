        subroutine iddr_qrpiv(m,n,a,krank,ind,ss)
c
c       computes the pivoted QR decomposition
c       of the matrix input into a, using Householder transformations,
c       _i.e._, transforms the matrix a from its input value in
c       to the matrix out with entry
c
c                               m
c       out(j,indprod(k))  =  Sigma  q(l,j) * in(l,k),
c                              l=1
c
c       for all j = 1, ..., krank, and k = 1, ..., n,
c
c       where in = the a from before the routine runs,
c       out = the a from after the routine runs,
c       out(j,k) = 0 when j > k (so that out is triangular),
c       q(1:m,1), ..., q(1:m,krank) are orthonormal,
c       indprod is the product of the permutations given by ind,
c       (as computable via the routine permmult,
c       with the permutation swapping 1 and ind(1) taken leftmost
c       in the product, that swapping 2 and ind(2) taken next leftmost,
c       ..., that swapping krank and ind(krank) taken rightmost),
c       and with the matrix out satisfying
c
c                  min(krank,m,n)
c       in(j,k)  =     Sigma      q(j,l) * out(l,indprod(k))
c                       l=1
c
c                +  epsilon(j,k),
c
c       for all j = 1, ..., m, and k = 1, ..., n,
c
c       for some matrix epsilon whose norm is (hopefully) minimized
c       by the pivoting procedure.
c       Well, technically, this routine outputs the Householder vectors
c       (or, rather, their second through last entries)
c       in the part of a that is supposed to get zeroed, that is,
c       in a(j,k) with m >= j > k >= 1.
c
c       input:
c       m -- first dimension of a and q
c       n -- second dimension of a
c       a -- matrix whose QR decomposition gets computed
c       krank -- desired rank of the output matrix
c                (please note that if krank > m or krank > n,
c                then the rank of the output matrix will be
c                less than krank)
c
c       output:
c       a -- triangular (R) factor in the QR decompositon
c            of the matrix input into the same storage locations, 
c            with the Householder vectors stored in the part of a
c            that would otherwise consist entirely of zeroes, that is,
c            in a(j,k) with m >= j > k >= 1
c       ind(k) -- index of the k^th pivot vector;
c                 the following code segment will correctly rearrange
c                 the product b of q and the upper triangle of out
c                 so that b best matches the input matrix in:
c
c                 copy the non-rearranged product of q and out into b
c                 set k to krank
c                 [start of loop]
c                   swap b(1:m,k) and b(1:m,ind(k))
c                   decrement k by 1
c                 if k > 0, then go to [start of loop]
c
c       work:
c       ss -- must be at least n real*8 words long
c
c       _N.B._: This routine outputs the Householder vectors
c       (or, rather, their second through last entries)
c       in the part of a that is supposed to get zeroed, that is,
c       in a(j,k) with m >= j > k >= 1.
c
c       reference:
c       Golub and Van Loan, "Matrix Computations," 3rd edition,
c            Johns Hopkins University Press, 1996, Chapter 5.
c
        implicit none
        integer n,m,ind(n),krank,k,j,kpiv,mm,nupdate,ifrescal,
     1          loops,loop
        real*8 a(m,n),ss(n),ssmax,scal,ssmaxin,rswap,feps
c
c
        feps = .1d-16
c
c
c       Compute the sum of squares of the entries in each column of a,
c       the maximum of all such sums, and find the first pivot
c       (column with the greatest such sum).
c
        ssmax = 0
        kpiv = 1
c
        do k = 1,n
c
          ss(k) = 0
          do j = 1,m
            ss(k) = ss(k)+a(j,k)**2
          enddo ! j
c
          if(ss(k) .gt. ssmax) then
            ssmax = ss(k)
            kpiv = k
          endif
c
        enddo ! k
c
        ssmaxin = ssmax
c
        nupdate = 0
c
c
c       Set loops = min(krank,m,n).
c
        loops = krank
        if(m .lt. loops) loops = m
        if(n .lt. loops) loops = n
c
        do loop = 1,loops
c
c
          mm = m-loop+1
c
c
c         Perform the pivoting.
c
          ind(loop) = kpiv
c
c         Swap a(1:m,loop) and a(1:m,kpiv).
c
          do j = 1,m
            rswap = a(j,loop)
            a(j,loop) = a(j,kpiv)
            a(j,kpiv) = rswap
          enddo ! j
c
c         Swap ss(loop) and ss(kpiv).
c
          rswap = ss(loop)
          ss(loop) = ss(kpiv)
          ss(kpiv) = rswap
c
c
          if(loop .lt. m) then
c
c
c           Compute the data for the Householder transformation
c           which will zero a(loop+1,loop), ..., a(m,loop)
c           when applied to a, replacing a(loop,loop)
c           with the first entry of the result of the application
c           of the Householder matrix to a(loop:m,loop),
c           and storing entries 2 to mm of the Householder vector
c           in a(loop+1,loop), ..., a(m,loop)
c           (which otherwise would get zeroed upon application
c           of the Householder transformation).
c
            call idd_house(mm,a(loop,loop),a(loop,loop),
     1                     a(loop+1,loop),scal)
            ifrescal = 0
c
c
c           Apply the Householder transformation
c           to the lower right submatrix of a
c           with upper leftmost entry at position (loop,loop+1).
c
            if(loop .lt. n) then
              do k = loop+1,n
                call idd_houseapp(mm,a(loop+1,loop),a(loop,k),
     1                            ifrescal,scal,a(loop,k))
              enddo ! k
            endif
c
c
c           Update the sums-of-squares array ss.
c
            do k = loop,n
              ss(k) = ss(k)-a(loop,k)**2
            enddo ! k
c
c
c           Find the pivot (column with the greatest sum of squares
c           of its entries).
c
            ssmax = 0
            kpiv = loop+1
c
            if(loop .lt. n) then
c
              do k = loop+1,n
c
                if(ss(k) .gt. ssmax) then
                  ssmax = ss(k)
                  kpiv = k
                endif
c
              enddo ! k
c
            endif ! loop .lt. n
c
c
c           Recompute the sums-of-squares and the pivot
c           when ssmax first falls below
c           sqrt((1000*feps)^2) * ssmaxin
c           and when ssmax first falls below
c           ((1000*feps)^2) * ssmaxin.
c
            if(
     1       (ssmax .lt. sqrt((1000*feps)**2) * ssmaxin
     2        .and. nupdate .eq. 0) .or.
     3       (ssmax .lt. ((1000*feps)**2) * ssmaxin
     4        .and. nupdate .eq. 1)
     5      ) then
c
              nupdate = nupdate+1
c
              ssmax = 0
              kpiv = loop+1
c
              if(loop .lt. n) then
c
                do k = loop+1,n
c
                  ss(k) = 0
                  do j = loop+1,m
                    ss(k) = ss(k)+a(j,k)**2
                  enddo ! j
c
                  if(ss(k) .gt. ssmax) then
                    ssmax = ss(k)
                    kpiv = k
                  endif
c
                enddo ! k
c
              endif ! loop .lt. n
c
            endif
c
c
          endif ! loop .lt. m
c
c
        enddo ! loop
c
c
        return
        end
