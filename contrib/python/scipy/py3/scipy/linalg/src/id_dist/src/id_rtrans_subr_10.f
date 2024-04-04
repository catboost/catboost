        subroutine idd_random_transf_init0(nsteps,n,albetas,ixs)
        implicit real *8 (a-h,o-z)
        save
        dimension albetas(2,n,*),ixs(n,*)
c
c       routine idd_random_transf_init serves as a memory wrapper
c       for the present routine; please see routine
c       idd_random_transf_init for documentation.
c
        do 2000 ijk=1,nsteps
c
        call idd_random_transf_init00(n,albetas(1,1,ijk),ixs(1,ijk) )
 2000 continue
        return
        end
c
c
c
c
c
