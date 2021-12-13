        subroutine idd_house(n,x,rss,vn,scal)
c
c       constructs the vector vn with vn(1) = 1
c       and the scalar scal such that
c       H := identity_matrix - scal * vn * transpose(vn) is orthogonal
c       and Hx = +/- e_1 * the root-sum-square of the entries of x
c       (H is the Householder matrix corresponding to x).
c
c       input:
c       n -- size of x and vn, though the indexing on vn goes
c            from 2 to n
c       x -- vector to reflect into its first component
c
c       output:
c       rss -- first entry of the vector resulting from the application
c              of the Householder matrix to x;
c              its absolute value is the root-sum-square
c              of the entries of x
c       vn -- entries 2 to n of the Householder vector vn;
c             vn(1) is assumed to be 1
c       scal -- scalar multiplying vn * transpose(vn);
c
c               scal = 2/(1 + vn(2)^2 + ... + vn(n)^2)
c               when vn(2), ..., vn(n) don't all vanish;
c
c               scal = 0
c               when vn(2), ..., vn(n) do all vanish
c               (including when n = 1)
c
c       reference:
c       Golub and Van Loan, "Matrix Computations," 3rd edition,
c            Johns Hopkins University Press, 1996, Chapter 5.
c
        implicit none
        save
        integer n,k
        real*8 x(n),rss,sum,v1,scal,vn(2:*),x1
c
c
        x1 = x(1)
c
c
c       Get out of this routine if n = 1.
c
        if(n .eq. 1) then
          rss = x1
          scal = 0
          return
        endif
c
c
c       Calculate (x(2))^2 + ... (x(n))^2
c       and the root-sum-square value of the entries in x.
c
c
        sum = 0
        do k = 2,n
          sum = sum+x(k)**2
        enddo ! k
c
c
c       Get out of this routine if sum = 0;
c       flag this case as such by setting v(2), ..., v(n) all to 0.
c
        if(sum .eq. 0) then
c
          rss = x1
          do k = 2,n
            vn(k) = 0
          enddo ! k
          scal = 0
c
          return
c
        endif
c
c
        rss = x1**2 + sum
        rss = sqrt(rss)
c
c
c       Determine the first component v1
c       of the unnormalized Householder vector
c       v = x - rss * (1 0 0 ... 0 0)^T.
c
c       If x1 <= 0, then form x1-rss directly,
c       since that expression cannot involve any cancellation.
c
        if(x1 .le. 0) v1 = x1-rss
c
c       If x1 > 0, then use the fact that
c       x1-rss = -sum / (x1+rss),
c       in order to avoid potential cancellation.
c
        if(x1 .gt. 0) v1 = -sum / (x1+rss)
c
c
c       Compute the vector vn and the scalar scal such that vn(1) = 1
c       in the Householder transformation
c       identity_matrix - scal * vn * transpose(vn).
c
        do k = 2,n
          vn(k) = x(k)/v1
        enddo ! k
c
c       scal = 2
c            / ( vn(1)^2 + vn(2)^2 + ... + vn(n)^2 )
c
c            = 2
c            / ( 1 + vn(2)^2 + ... + vn(n)^2 )
c
c            = 2*v(1)^2
c            / ( v(1)^2 + (v(1)*vn(2))^2 + ... + (v(1)*vn(n))^2 )
c
c            = 2*v(1)^2
c            / ( v(1)^2 + (v(2)^2 + ... + v(n)^2) )
c
        scal = 2*v1**2 / (v1**2+sum)
c
c
        return
        end
c
c
c
c
