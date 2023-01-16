        subroutine idzp_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
     1                        matvec,p1,p2,p3,p4,krank,u,v,s,ier,
     2                        list,proj,col,work)
c
c       routine idzp_rsvd serves as a memory wrapper
c       for the present routine (please see routine idzp_rsvd
c       for further documentation).
c
        implicit none
        integer m,n,krank,list(n),ier
        real*8 s(krank)
        complex*16 p1t,p2t,p3t,p4t,p1,p2,p3,p4,u(m,krank),v(n,krank),
     1             proj(krank,n-krank),col(m*krank),
     2             work((krank+1)*(m+3*n+10)+9*krank**2)
        external matveca,matvec
c
c
c       Collect together the columns of a indexed by list into col.
c
        call idz_getcols(m,n,matvec,p1,p2,p3,p4,krank,list,col,work)
c
c
c       Convert the ID to an SVD.
c
        call idz_id2svd(m,krank,col,n,list,proj,u,v,s,ier,work)
c
c
        return
        end
c
c
c
c
