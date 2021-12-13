        subroutine iddp_rid(lproj,eps,m,n,matvect,p1,p2,p3,p4,
     1                      krank,list,proj,ier)
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
c       for some matrix epsilon dimensioned epsilon(m,n-krank)
c       such that the greatest singular value of epsilon
c       <= the greatest singular value of a * eps.
c
c       input:
c       lproj -- maximum usable length (in real*8 elements)
c                of the array proj
c       eps -- precision to which the ID is to be computed
c       m -- first dimension of a
c       n -- second dimension of a
c       matvect -- routine which applies the transpose
c                  of the matrix to be ID'd to an arbitrary vector;
c                  this routine must have a calling sequence
c                  of the form
c
c                  matvect(m,x,n,y,p1,p2,p3,p4),
c
c                  where m is the length of x,
c                  x is the vector to which the transpose
c                  of the matrix is to be applied,
c                  n is the length of y,
c                  y is the product of the transposed matrix and x,
c                  and p1, p2, p3, and p4 are user-specified parameters
c       p1 -- parameter to be passed to routine matvect
c       p2 -- parameter to be passed to routine matvect
c       p3 -- parameter to be passed to routine matvect
c       p4 -- parameter to be passed to routine matvect
c
c       output:
c       krank -- numerical rank
c       list -- indices of the columns in the ID
c       proj -- matrix of coefficients needed to interpolate
c               from the selected columns to the other columns
c               in the original matrix being ID'd;
c               the present routine uses proj as a work array, too, so
c               proj must be at least m+1 + 2*n*(krank+1) real*8
c               elements long, where krank is the rank output
c               by the present routine
c       ier -- 0 when the routine terminates successfully;
c              -1000 when lproj is too small
c
c       _N.B._: The algorithm used by this routine is randomized.
c               proj must be at least m+1 + 2*n*(krank+1) real*8
c               elements long, where krank is the rank output
c               by the present routine.
c
c       reference:
c       Halko, Martinsson, Tropp, "Finding structure with randomness:
c            probabilistic algorithms for constructing approximate
c            matrix decompositions," SIAM Review, 53 (2): 217-288,
c            2011.
c
        implicit none
        integer m,n,list(n),krank,lw,iwork,lwork,ira,kranki,lproj,
     1          lra,ier,k
        real*8 eps,p1,p2,p3,p4,proj(*)
        external matvect
c
c
        ier = 0
c
c
c       Allocate memory in proj.
c
        lw = 0
c
        iwork = lw+1
        lwork = m+2*n+1
        lw = lw+lwork
c
        ira = lw+1
c
c
c       Find the rank of a.
c
        lra = lproj-lwork
        call idd_findrank(lra,eps,m,n,matvect,p1,p2,p3,p4,
     1                    kranki,proj(ira),ier,proj(iwork))
        if(ier .ne. 0) return
c
c
        if(lproj .lt. lwork+2*kranki*n) then
          ier = -1000
          return
        endif
c
c
c       Transpose ra.
c
        call idd_rtransposer(n,kranki,proj(ira),proj(ira+kranki*n))
c
c
c       Move the tranposed matrix to the beginning of proj.
c
        do k = 1,kranki*n
          proj(k) = proj(ira+kranki*n+k-1)
        enddo ! k
c
c
c       ID the transposed matrix.
c
        call iddp_id(eps,kranki,n,proj,krank,list,proj(1+kranki*n))
c
c
        return
        end
c
c
c
c
