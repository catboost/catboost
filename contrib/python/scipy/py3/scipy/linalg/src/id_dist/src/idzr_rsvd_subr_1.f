        subroutine idzr_rsvd0(m,n,matveca,p1t,p2t,p3t,p4t,
     1                        matvec,p1,p2,p3,p4,krank,u,v,s,ier,
     2                        list,proj,col,work)
c
c       routine idzr_rsvd serves as a memory wrapper
c       for the present routine (please see routine idzr_rsvd
c       for further documentation).
c
        implicit none
        integer m,n,krank,list(n),ier,k
        real*8 s(krank)
        complex*16 p1t,p2t,p3t,p4t,p1,p2,p3,p4,u(m,krank),v(n,krank),
     1             proj(krank*(n-krank)),col(m*krank),
     2             work((krank+1)*(m+3*n+10)+9*krank**2)
        external matveca,matvec
c
c
c       ID a.
c
        call idzr_rid(m,n,matveca,p1t,p2t,p3t,p4t,krank,list,work)
c
c
c       Retrieve proj from work.
c
        do k = 1,krank*(n-krank)
          proj(k) = work(k)
        enddo ! k
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
