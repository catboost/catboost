        subroutine iddr_id(m,n,a,krank,list,rnorms)
c
c       computes the ID of a, i.e., lists in list the indices
c       of krank columns of a such that 
c
c       a(j,list(k))  =  a(j,list(k))
c
c       for all j = 1, ..., m; k = 1, ..., krank, and
c
c                        krank
c       a(j,list(k))  =  Sigma  a(j,list(l)) * proj(l,k-krank)       (*)
c                         l=1
c
c                     +  epsilon(j,k-krank)
c
c       for all j = 1, ..., m; k = krank+1, ..., n,
c
c       for some matrix epsilon, dimensioned epsilon(m,n-krank),
c       whose norm is (hopefully) minimized by the pivoting procedure.
c       The present routine stores the krank x (n-krank) matrix proj
c       in the memory initially occupied by a.
c
c       input:
c       m -- first dimension of a
c       n -- second dimension of a, as well as the dimension required
c            of list
c       a -- matrix to be ID'd
c       krank -- desired rank of the output matrix
c                (please note that if krank > m or krank > n,
c                then the rank of the output matrix will be
c                less than krank)
c
c       output:
c       a -- the first krank*(n-krank) elements of a constitute
c            the krank x (n-krank) interpolation matrix proj
c       list -- list of the indices of the krank columns of a
c               through which the other columns of a are expressed;
c               also, list describes the permutation of proj
c               required to reconstruct a as indicated in (*) above
c       rnorms -- absolute values of the entries on the diagonal
c                 of the triangular matrix used to compute the ID
c                 (these may be used to check the stability of the ID)
c
c       _N.B._: This routine changes a.
c
c       reference:
c       Cheng, Gimbutas, Martinsson, Rokhlin, "On the compression of
c            low-rank matrices," SIAM Journal on Scientific Computing,
c            26 (4): 1389-1404, 2005.
c
        implicit none
        integer m,n,krank,j,k,list(n),iswap
        real*8 a(m,n),rnorms(n),ss
c
c
c       QR decompose a.
c
        call iddr_qrpiv(m,n,a,krank,list,rnorms)
c
c
c       Build the list of columns chosen in a
c       by multiplying together the permutations in list,
c       with the permutation swapping 1 and list(1) taken rightmost
c       in the product, that swapping 2 and list(2) taken next
c       rightmost, ..., that swapping krank and list(krank) taken
c       leftmost.
c
        do k = 1,n
          rnorms(k) = k
        enddo ! k
c
        if(krank .gt. 0) then
          do k = 1,krank
c
c           Swap rnorms(k) and rnorms(list(k)).
c
            iswap = rnorms(k)
            rnorms(k) = rnorms(list(k))
            rnorms(list(k)) = iswap
c
          enddo ! k
        endif
c
        do k = 1,n
          list(k) = rnorms(k)
        enddo ! k
c
c
c       Fill rnorms for the output.
c
        ss = 0
c
        do k = 1,krank
          rnorms(k) = a(k,k)
          ss = ss+rnorms(k)**2
        enddo ! k
c
c
c       Backsolve for proj, storing it at the beginning of a.
c
        if(krank .gt. 0 .and. ss .gt. 0) then
          call idd_lssolve(m,n,a,krank)
        endif
c
        if(ss .eq. 0) then
c
          do k = 1,n
            do j = 1,m
c
              a(j,k) = 0
c
            enddo ! j
          enddo ! k
c
        endif
c
c
        return
        end
c
c
c
c
