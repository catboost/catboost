        subroutine idzp_asvd0(m,n,a,krank,list,proj,u,v,s,ier,
     1                        col,work)
c
c       routine idzp_asvd serves as a memory wrapper
c       for the present routine (please see routine idzp_asvd
c       for further documentation).
c
        implicit none
        integer m,n,krank,list(n),ier
        real*8 s(krank)
        complex*16 a(m,n),u(m,krank),v(n,krank),
     1             proj(krank,n-krank),col(m,krank),
     2             work((krank+1)*(m+3*n+10)+9*krank**2)
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
c
c
c
c
