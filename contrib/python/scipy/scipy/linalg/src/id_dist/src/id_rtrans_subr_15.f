        subroutine idz_random_transf00(x,y,n,albetas,gammas,ixs)
        implicit real *8 (a-h,o-z)
        save
        complex *16 x(*),y(*),gammas(*),a,b
        dimension albetas(2,*),ixs(*)
c
c       implements one step of the random transform
c       required by routine idz_random_transf0 (please see the latter).
c
c        implement the permutation
c        and multiply by the random numbers
c        on the unit circle
c
        do 1600 i=1,n
c
        j=ixs(i)
        y(i)=x(j)*gammas(i)
 1600 continue
c
c        implement 2 \times 2 matrices
c
        do 2600 i=1,n-1
c
        alpha=albetas(1,i)
        beta=albetas(2,i)
c
        a=y(i)
        b=y(i+1)
c
        y(i)=alpha*a+beta*b
        y(i+1)=-beta*a+alpha*b
 2600 continue
c
        return
        end
c
c
c
c
c
