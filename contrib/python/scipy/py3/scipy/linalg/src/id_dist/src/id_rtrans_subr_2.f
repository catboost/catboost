        subroutine idd_random_transf00_inv(x,y,n,albetas,ixs)
        implicit real *8 (a-h,o-z)
        save
        dimension x(*),y(*),albetas(2,*),ixs(*)
c
c       implements one step of the random transform required
c       by routine idd_random_transf0_inv (please see the latter).
c
c
c        implement 2 \times 2 matrices
c
        do 1600 i=1,n
        y(i)=x(i)
 1600 continue
c
        do 1800 i=n-1,1,-1
c
        alpha=albetas(1,i)
        beta=albetas(2,i)
c
        a=y(i)
        b=y(i+1)
c
        y(i)=alpha*a-beta*b
        y(i+1)=beta*a+alpha*b
 1800 continue
c
c        implement the permutation
c
        do 2600 i=1,n
c
        j=ixs(i)
        x(j)=y(i)
 2600 continue
c
        do 2800 i=1,n
c
        y(i)=x(i)
 2800 continue
c
        return
        end
c
c
c
c
c
