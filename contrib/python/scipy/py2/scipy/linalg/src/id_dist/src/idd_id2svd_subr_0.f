        subroutine idd_id2svd(m,krank,b,n,list,proj,u,v,s,ier,w)
c
c       converts an approximation to a matrix in the form of an ID
c       to an approximation in the form of an SVD.
c
c       input:
c       m -- first dimension of b
c       krank -- rank of the ID
c       b -- columns of the original matrix in the ID
c       list -- list of columns chosen from the original matrix
c               in the ID
c       n -- length of list and part of the second dimension of proj
c       proj -- projection coefficients in the ID
c
c       output:
c       u -- left singular vectors
c       v -- right singular vectors
c       s -- singular values
c       ier -- 0 when the routine terminates successfully;
c              nonzero otherwise
c
c       work:
c       w -- must be at least (krank+1)*(m+3*n)+26*krank**2 real*8
c            elements long
c
c       _N.B._: This routine destroys b.
c
        implicit none
        integer m,krank,n,list(n),iwork,lwork,ip,lp,it,lt,ir,lr,
     1          ir2,lr2,ir3,lr3,iind,lind,iindt,lindt,lw,ier
        real*8 b(m,krank),proj(krank,n-krank),u(m,krank),v(n,krank),
     1         w((krank+1)*(m+3*n)+26*krank**2),s(krank)
c
c
        lw = 0
c
        iwork = lw+1
        lwork = 25*krank**2
        lw = lw+lwork
c
        ip = lw+1
        lp = krank*n
        lw = lw+lp
c
        it = lw+1
        lt = n*krank
        lw = lw+lt
c
        ir = lw+1
        lr = krank*n
        lw = lw+lr
c
        ir2 = lw+1
        lr2 = krank*m
        lw = lw+lr2
c
        ir3 = lw+1
        lr3 = krank*krank
        lw = lw+lr3
c
        iind = lw+1
        lind = n/2+1
        lw = lw+1
c
        iindt = lw+1
        lindt = m/2+1
        lw = lw+1
c
c
        call idd_id2svd0(m,krank,b,n,list,proj,u,v,s,ier,
     1                   w(iwork),w(ip),w(it),w(ir),w(ir2),w(ir3),
     2                   w(iind),w(iindt))
c
c
        return
        end
c
c
c
c
