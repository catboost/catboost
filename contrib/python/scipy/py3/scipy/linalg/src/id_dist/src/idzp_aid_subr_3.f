        subroutine idz_estrank(eps,m,n,a,w,krank,ra)
c
c       estimates the numerical rank krank of an m x n matrix a
c       to precision eps. This routine applies n2 random vectors
c       to a, obtaining ra, where n2 is the greatest integer
c       less than or equal to m such that n2 is a positive integer
c       power of two. krank is typically about 8 higher than
c       the actual numerical rank.
c
c       input:
c       eps -- precision defining the numerical rank
c       m -- first dimension of a
c       n -- second dimension of a
c       a -- matrix whose rank is to be estimated
c       w -- initialization array that has been constructed
c            by routine idz_frmi
c
c       output:
c       krank -- estimate of the numerical rank of a;
c                this routine returns krank = 0 when the actual
c                numerical rank is nearly full (that is,
c                greater than n - 8 or n2 - 8)
c       ra -- product of an n2 x m random matrix and the m x n matrix
c             a, where n2 is the greatest integer less than or equal
c             to m such that n2 is a positive integer power of two;
c             ra doubles as a work array in the present routine, and so
c             must be at least n*n2+(n+1)*(n2+1) complex*16 elements
c             long
c
c       _N.B._: ra must be at least n*n2+(n2+1)*(n+1) complex*16
c               elements long for use in the present routine
c               (here, n2 is the greatest integer less than or equal
c               to m, such that n2 is a positive integer power of two).
c               This routine returns krank = 0 when the actual
c               numerical rank is nearly full.
c
        implicit none
        integer m,n,krank,n2,irat,lrat,iscal,lscal,ira,lra,lra2
        real*8 eps
        complex*16 a(m,n),ra(*),w(17*m+70)
c
c
c       Extract from the array w initialized by routine idz_frmi
c       the greatest integer less than or equal to m that is
c       a positive integer power of two.
c
        n2 = w(2)
c
c
c       Allocate memory in ra.
c
        lra = 0
c
        ira = lra+1
        lra2 = n2*n
        lra = lra+lra2
c
        irat = lra+1
        lrat = n*(n2+1)
        lra = lra+lrat
c
        iscal = lra+1
        lscal = n2+1
        lra = lra+lscal
c
        call idz_estrank0(eps,m,n,a,w,n2,krank,ra(ira),ra(irat),
     1                    ra(iscal))
c
c
        return
        end
c
c
c
c
