        subroutine idd_findrank(lra,eps,m,n,matvect,p1,p2,p3,p4,
     1                          krank,ra,ier,w)
c
c       estimates the numerical rank krank of a matrix a to precision
c       eps, where the routine matvect applies the transpose of a
c       to an arbitrary vector. This routine applies the transpose of a
c       to krank random vectors, and returns the resulting vectors
c       as the columns of ra.
c
c       input:
c       lra -- maximum usable length (in real*8 elements) of array ra
c       eps -- precision defining the numerical rank
c       m -- first dimension of a
c       n -- second dimension of a
c       matvect -- routine which applies the transpose
c                  of the matrix whose rank is to be estimated
c                  to an arbitrary vector; this routine must have
c                  a calling sequence of the form
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
c       krank -- estimate of the numerical rank of a
c       ra -- product of the transpose of a and a matrix whose entries
c             are pseudorandom realizations of i.i.d. random numbers,
c             uniformly distributed on [0,1];
c             ra must be at least 2*n*krank real*8 elements long
c       ier -- 0 when the routine terminates successfully;
c              -1000 when lra is too small
c
c       work:
c       w -- must be at least m+2*n+1 real*8 elements long
c
c       _N.B._: ra must be at least 2*n*krank real*8 elements long.
c               Also, the algorithm used by this routine is randomized.
c
        implicit none
        integer m,n,lw,krank,ix,lx,iy,ly,iscal,lscal,lra,ier
        real*8 eps,p1,p2,p3,p4,ra(n,*),w(m+2*n+1)
        external matvect
c
c
        lw = 0
c
        ix = lw+1
        lx = m
        lw = lw+lx
c
        iy = lw+1
        ly = n
        lw = lw+ly
c
        iscal = lw+1
        lscal = n+1
        lw = lw+lscal
c
c
        call idd_findrank0(lra,eps,m,n,matvect,p1,p2,p3,p4,
     1                     krank,ra,ier,w(ix),w(iy),w(iscal))
c
c
        return
        end
c
c
c
c
