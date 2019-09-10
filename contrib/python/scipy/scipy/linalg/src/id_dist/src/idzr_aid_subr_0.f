        subroutine idzr_aid(m,n,a,krank,w,list,proj)
c
c       computes the ID of the matrix a, i.e., lists in list
c       the indices of krank columns of a such that 
c
c       a(j,list(k))  =  a(j,list(k))
c
c       for all j = 1, ..., m; k = 1, ..., krank, and
c
c                       min(m,n,krank)
c       a(j,list(k))  =     Sigma      a(j,list(l)) * proj(l,k-krank)(*)
c                            l=1
c
c                     +  epsilon(j,k-krank)
c
c       for all j = 1, ..., m; k = krank+1, ..., n,
c
c       for some matrix epsilon, dimensioned epsilon(m,n-krank),
c       whose norm is (hopefully) minimized by the pivoting procedure.
c
c       input:
c       m -- number of rows in a
c       n -- number of columns in a
c       a -- matrix to be ID'd; the present routine does not alter a
c       krank -- rank of the ID to be constructed
c       w -- initialization array that routine idzr_aidi
c            has constructed
c
c       output:
c       list -- indices of the columns in the ID
c       proj -- matrix of coefficients needed to interpolate
c               from the selected columns to the other columns
c               in the original matrix being ID'd
c
c       _N.B._: The algorithm used by this routine is randomized.
c
c       reference:
c       Halko, Martinsson, Tropp, "Finding structure with randomness:
c            probabilistic algorithms for constructing approximate
c            matrix decompositions," SIAM Review, 53 (2): 217-288,
c            2011.
c
        implicit none
        integer m,n,krank,list(n),lw,ir,lr,lw2,iw
        complex*16 a(m,n),proj(krank*(n-krank)),
     1             w((2*krank+17)*n+21*m+80)
c
c
c       Allocate memory in w.
c
        lw = 0
c
        iw = lw+1
        lw2 = 21*m+80+n
        lw = lw+lw2
c
        ir = lw+1
        lr = (krank+8)*2*n
        lw = lw+lr
c
c
        call idzr_aid0(m,n,a,krank,w(iw),list,proj,w(ir))
c
c
        return
        end
c
c
c
c
