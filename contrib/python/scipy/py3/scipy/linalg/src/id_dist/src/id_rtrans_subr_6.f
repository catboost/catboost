        subroutine idd_random_transf00(x,y,n,albetas,ixs)
        implicit real *8 (a-h,o-z)
        save
        dimension x(*),y(*),albetas(2,*),ixs(*)
c
c       implements one step of the random transform
c       required by routine idd_random_transf0 (please see the latter).
c
c        implement the permutation
c
        do 1600 i=1,n
c
        j=ixs(i)
        y(i)=x(j)
 1600 continue
c
c        implement 2 \times 2 matrices
c
        do 1800 i=1,n-1
c
        alpha=albetas(1,i)
        beta=albetas(2,i)
c
        a=y(i)
        b=y(i+1)
c
        y(i)=alpha*a+beta*b
        y(i+1)=-beta*a+alpha*b
 1800 continue
c
        return
        end
c
c
c
c
c
