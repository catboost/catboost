        subroutine idz_random_transf_init00(n,albetas,gammas,ixs)
        implicit real *8 (a-h,o-z)
        save
        dimension albetas(2,*),gammas(*),ixs(*)
c
c       constructs one stage of the random transform
c       initialized by routine idz_random_transf_init0
c       (please see the latter).
c
        done=1
        twopi=2*4*atan(done)
c
c        construct the random permutation
c
        ifrepeat=0
        call id_randperm(n,ixs)
c
c        construct the random variables
c
        call id_srand(2*n,albetas)
        call id_srand(2*n,gammas)
c
        do 1300 i=1,n
c
        albetas(1,i)=2*albetas(1,i)-1
        albetas(2,i)=2*albetas(2,i)-1
        gammas(2*i-1)=2*gammas(2*i-1)-1
        gammas(2*i)=2*gammas(2*i)-1
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
c
c        construct the random multipliers on the unit circle
c
        do 1500 i=1,n
c
        d=gammas(2*i-1)**2+gammas(2*i)**2
        d=1/sqrt(d)
c
c        fill the real part
c
        gammas(2*i-1)=gammas(2*i-1)*d
c
c        fill the imaginary part
c
        gammas(2*i)=gammas(2*i)*d
 1500 continue
c
        return
        end
c
c
c
c
c
