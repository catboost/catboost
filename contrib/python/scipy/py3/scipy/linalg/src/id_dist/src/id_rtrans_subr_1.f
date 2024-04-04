        subroutine idd_random_transf0_inv(nsteps,x,y,n,w2,albetas,iixs)
        implicit real *8 (a-h,o-z)
        save
        dimension x(*),y(*),w2(*),albetas(2,n,*),iixs(n,*)
c
c       routine idd_random_transf_inverse serves as a memory wrapper
c       for the present routine; see routine idd_random_transf_inverse
c       for documentation.
c
        do 1200 i=1,n
c
        w2(i)=x(i)
 1200 continue
c
        do 2000 ijk=nsteps,1,-1
c
        call idd_random_transf00_inv(w2,y,n,albetas(1,1,ijk),
     1      iixs(1,ijk) )
c
        do 1400 j=1,n
c
        w2(j)=y(j)
 1400 continue
 2000 continue
c
        return
        end
c
c
c
c
c
