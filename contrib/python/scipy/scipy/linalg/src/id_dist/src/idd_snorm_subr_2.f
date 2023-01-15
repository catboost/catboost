        subroutine idd_diffsnorm(m,n,matvect,p1t,p2t,p3t,p4t,
     1                           matvect2,p1t2,p2t2,p3t2,p4t2,
     2                           matvec,p1,p2,p3,p4,
     3                           matvec2,p12,p22,p32,p42,its,snorm,w)
c
c       estimates the spectral norm of the difference between matrices
c       a and a2, where a is specified by routines matvec and matvect
c       for applying a and a^T to arbitrary vectors,
c       and a2 is specified by routines matvec2 and matvect2
c       for applying a2 and (a2)^T to arbitrary vectors.
c       This routine uses the power method
c       with a random starting vector.
c
c       input:
c       m -- number of rows in a, as well as the number of rows in a2
c       n -- number of columns in a, as well as the number of columns
c            in a2
c       matvect -- routine which applies the transpose of a
c                  to an arbitrary vector; this routine must have
c                  a calling sequence of the form
c
c                  matvect(m,x,n,y,p1t,p2t,p3t,p4t),
c
c                  where m is the length of x,
c                  x is the vector to which the transpose of a
c                  is to be applied,
c                  n is the length of y,
c                  y is the product of the transpose of a and x,
c                  and p1t, p2t, p3t, and p4t are user-specified
c                  parameters
c       p1t -- parameter to be passed to routine matvect
c       p2t -- parameter to be passed to routine matvect
c       p3t -- parameter to be passed to routine matvect
c       p4t -- parameter to be passed to routine matvect
c       matvect2 -- routine which applies the transpose of a2
c                   to an arbitrary vector; this routine must have
c                   a calling sequence of the form
c
c                   matvect2(m,x,n,y,p1t2,p2t2,p3t2,p4t2),
c
c                   where m is the length of x,
c                   x is the vector to which the transpose of a2
c                   is to be applied,
c                   n is the length of y,
c                   y is the product of the transpose of a2 and x,
c                   and p1t2, p2t2, p3t2, and p4t2 are user-specified
c                   parameters
c       p1t2 -- parameter to be passed to routine matvect2
c       p2t2 -- parameter to be passed to routine matvect2
c       p3t2 -- parameter to be passed to routine matvect2
c       p4t2 -- parameter to be passed to routine matvect2
c       matvec -- routine which applies the matrix a
c                 to an arbitrary vector; this routine must have
c                 a calling sequence of the form
c
c                 matvec(n,x,m,y,p1,p2,p3,p4),
c
c                 where n is the length of x,
c                 x is the vector to which a is to be applied,
c                 m is the length of y,
c                 y is the product of a and x,
c                 and p1, p2, p3, and p4 are user-specified parameters
c       p1 -- parameter to be passed to routine matvec
c       p2 -- parameter to be passed to routine matvec
c       p3 -- parameter to be passed to routine matvec
c       p4 -- parameter to be passed to routine matvec
c       matvec2 -- routine which applies the matrix a2
c                  to an arbitrary vector; this routine must have
c                  a calling sequence of the form
c
c                  matvec2(n,x,m,y,p12,p22,p32,p42),
c
c                  where n is the length of x,
c                  x is the vector to which a2 is to be applied,
c                  m is the length of y,
c                  y is the product of a2 and x, and
c                  p12, p22, p32, and p42 are user-specified parameters
c       p12 -- parameter to be passed to routine matvec2
c       p22 -- parameter to be passed to routine matvec2
c       p32 -- parameter to be passed to routine matvec2
c       p42 -- parameter to be passed to routine matvec2
c       its -- number of iterations of the power method to conduct
c
c       output:
c       snorm -- estimate of the spectral norm of a-a2
c
c       work:
c       w -- must be at least 3*m+3*n real*8 elements long
c
c       reference:
c       Kuczynski and Wozniakowski, "Estimating the largest eigenvalue
c            by the power and Lanczos algorithms with a random start,"
c            SIAM Journal on Matrix Analysis and Applications,
c            13 (4): 1992, 1094-1122.
c
        implicit none
        integer m,n,its,lw,iu,lu,iu1,lu1,iu2,lu2,
     1          iv,lv,iv1,lv1,iv2,lv2
        real*8 snorm,p1t,p2t,p3t,p4t,p1t2,p2t2,p3t2,p4t2,
     1         p1,p2,p3,p4,p12,p22,p32,p42,w(3*m+3*n)
        external matvect,matvec,matvect2,matvec2
c
c
c       Allocate memory in w.
c
        lw = 0
c
        iu = lw+1
        lu = m
        lw = lw+lu
c
        iu1 = lw+1
        lu1 = m
        lw = lw+lu1
c
        iu2 = lw+1
        lu2 = m
        lw = lw+lu2
c
        iv = lw+1
        lv = n
        lw = lw+1
c
        iv1 = lw+1
        lv1 = n
        lw = lw+lv1
c
        iv2 = lw+1
        lv2 = n
        lw = lw+lv2
c
c
        call idd_diffsnorm0(m,n,matvect,p1t,p2t,p3t,p4t,
     1                      matvect2,p1t2,p2t2,p3t2,p4t2,
     2                      matvec,p1,p2,p3,p4,
     3                      matvec2,p12,p22,p32,p42,
     4                      its,snorm,w(iu),w(iu1),w(iu2),
     5                      w(iv),w(iv1),w(iv2))
c
c
        return
        end
c
c
c
c
