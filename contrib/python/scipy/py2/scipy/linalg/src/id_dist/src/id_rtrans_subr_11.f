        subroutine idd_random_transf_init00(n,albetas,ixs)
        implicit real *8 (a-h,o-z)
        save
        dimension albetas(2,*),ixs(*)
c
c       constructs one stage of the random transform
c       initialized by routine idd_random_transf_init0
c       (please see the latter).
c
c        construct the random permutation
c
        ifrepeat=0
        call id_randperm(n,ixs)
c
c        construct the random variables
c
        call id_srand(2*n,albetas)
c
        do 1300 i=1,n
c
        albetas(1,i)=2*albetas(1,i)-1
        albetas(2,i)=2*albetas(2,i)-1
 1300 continue
c
c        construct the random 2 \times 2 transformations
c
        do 1400 i=1,n
c
        d=albetas(1,i)**2+albetas(2,i)**2
        d=1/sqrt(d)
        albetas(1,i)=albetas(1,i)*d
        albetas(2,i)=albetas(2,i)*d
 1400 continue
        return
        end
