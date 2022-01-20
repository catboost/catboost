        subroutine iddr_asvd0(m,n,a,krank,winit,u,v,s,ier,
     1                        list,proj,col,work)
c
c       routine iddr_asvd serves as a memory wrapper
c       for the present routine (please see routine iddr_asvd
c       for further documentation).
c
        implicit none
        integer m,n,krank,list(n),ier
        real*8 a(m,n),u(m,krank),v(n,krank),s(krank),
     1         proj(krank,n-krank),col(m*krank),
     2         winit((2*krank+17)*n+27*m+100),
     3         work((krank+1)*(m+3*n)+26*krank**2)
c
c
c       ID a.
c
        call iddr_aid(m,n,a,krank,winit,list,proj)
c
c
c       Collect together the columns of a indexed by list into col.
c
        call idd_copycols(m,n,a,krank,list,col)
c
c
c       Convert the ID to an SVD.
c
        call idd_id2svd(m,krank,col,n,list,proj,u,v,s,ier,work)
c
c
        return
        end
