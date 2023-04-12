        subroutine idzr_asvd0(m,n,a,krank,winit,u,v,s,ier,
     1                        list,proj,col,work)
c
c       routine idzr_asvd serves as a memory wrapper
c       for the present routine (please see routine idzr_asvd
c       for further documentation).
c
        implicit none
        integer m,n,krank,list(n),ier
        real*8 s(krank)
        complex*16 a(m,n),u(m,krank),v(n,krank),
     1             proj(krank,n-krank),col(m*krank),
     2             winit((2*krank+17)*n+21*m+80),
     3             work((krank+1)*(m+3*n+10)+9*krank**2)
c
c
c       ID a.
c
        call idzr_aid(m,n,a,krank,winit,list,proj)
c
c
c       Collect together the columns of a indexed by list into col.
c
        call idz_copycols(m,n,a,krank,list,col)
c
c
c       Convert the ID to an SVD.
c
        call idz_id2svd(m,krank,col,n,list,proj,u,v,s,ier,work)
c
c
        return
        end
