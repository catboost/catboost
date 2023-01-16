        subroutine idzp_aid(eps,m,n,a,work,krank,list,proj)
c
c       computes the ID of the matrix a, i.e., lists in list
c       the indices of krank columns of a such that
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
c       for some matrix epsilon dimensioned epsilon(m,n-krank)
c       such that the greatest singular value of epsilon
c       <= the greatest singular value of a * eps.
c
c       input:
c       eps -- precision to which the ID is to be computed
c       m -- first dimension of a
c       n -- second dimension of a
c       a -- matrix to be decomposed; the present routine does not
c            alter a
c       work -- initialization array that has been constructed
c               by routine idz_frmi
c
c       output:
c       krank -- numerical rank of a to precision eps
c       list -- indices of the columns in the ID
c       proj -- matrix of coefficients needed to interpolate
c               from the selected columns to the other columns
c               in the original matrix being ID'd;
c               proj doubles as a work array in the present routine, so
c               proj must be at least n*(2*n2+1)+n2+1 complex*16
c               elements long, where n2 is the greatest integer
c               less than or equal to m, such that n2 is
c               a positive integer power of two.
c
c       _N.B._: The algorithm used by this routine is randomized.
c               proj must be at least n*(2*n2+1)+n2+1 complex*16
c               elements long, where n2 is the greatest integer
c               less than or equal to m, such that n2 is
c               a positive integer power of two.
c
c       reference:
c       Halko, Martinsson, Tropp, "Finding structure with randomness:
c            probabilistic algorithms for constructing approximate
c            matrix decompositions," SIAM Review, 53 (2): 217-288,
c            2011.
c
        implicit none
        integer m,n,list(n),krank,kranki,n2
        real*8 eps
        complex*16 a(m,n),proj(*),work(17*m+70)
c
c
c       Allocate memory in proj.
c
        n2 = work(2)
c
c
c       Find the rank of a.
c
        call idz_estrank(eps,m,n,a,work,kranki,proj)
c
c
        if(kranki .eq. 0) call idzp_aid0(eps,m,n,a,krank,list,proj,
     1                                   proj(m*n+1))
c
        if(kranki .ne. 0) call idzp_aid1(eps,n2,n,kranki,proj,
     1                                   krank,list,proj(n2*n+1))
c
c
        return
        end
c
c
c
c
