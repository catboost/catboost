        subroutine idz_random_transf00_inv(x,y,n,albetas,gammas,ixs)
        implicit real *8 (a-h,o-z)
        save
        complex *16 x(*),y(*),gammas(*),a,b
        dimension albetas(2,*),ixs(*)
c
c       implements one step of the random transform
c       required by routine idz_random_transf0_inv
c       (please see the latter).
c
c        implement 2 \times 2 matrices
c
        do 1600 i=n-1,1,-1
c
        alpha=albetas(1,i)
        beta=albetas(2,i)
c
        a=x(i)
        b=x(i+1)
c
        x(i)=alpha*a-beta*b
        x(i+1)=beta*a+alpha*b
 1600 continue
c
c        implement the permutation
c        and divide by the random numbers on the unit circle
c        (or, equivalently, multiply by their conjugates)
c
        do 1800 i=1,n
c
        j=ixs(i)
        y(j)=x(i)*conjg(gammas(i))
 1800 continue
c
        return
        end
c
c
c
c
c
