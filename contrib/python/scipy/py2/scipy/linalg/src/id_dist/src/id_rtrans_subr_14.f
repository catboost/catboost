        subroutine idz_random_transf0(nsteps,x,y,n,w2,albetas,
     1      gammas,iixs)
        implicit real *8 (a-h,o-z)
        save
        complex *16 x(*),y(*),w2(*),gammas(n,*)
        dimension albetas(2,n,*),iixs(n,*)
c
c       routine idz_random_transf serves as a memory wrapper
c       for the present routine; please see routine idz_random_transf
c       for documentation.
c
        do 1200 i=1,n
c
        w2(i)=x(i)
 1200 continue
c
        do 2000 ijk=1,nsteps
c
        call idz_random_transf00(w2,y,n,albetas(1,1,ijk),
     1      gammas(1,ijk),iixs(1,ijk) )
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
