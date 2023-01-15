        subroutine idz_house(n,x,css,vn,scal)
c
c       constructs the vector vn with vn(1) = 1,
c       and the scalar scal, such that the obviously self-adjoint
c       H := identity_matrix - scal * vn * adjoint(vn) is unitary,
c       the absolute value of the first entry of Hx
c       is the root-sum-square of the entries of x,
c       and all other entries of Hx are zero
c       (H is the Householder matrix corresponding to x).
c
c       input:
c       n -- size of x and vn, though the indexing on vn goes
c            from 2 to n
c       x -- vector to reflect into its first component
c
c       output:
c       css -- root-sum-square of the entries of x * the phase of x(1)
c       vn -- entries 2 to n of the Householder vector vn;
c             vn(1) is assumed to be 1
c       scal -- scalar multiplying vn * adjoint(vn);
c
c               scal = 2/(1 + |vn(2)|^2 + ... + |vn(n)|^2)
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
        real*8 scal,test,rss,sum
        complex*16 x(n),v1,vn(2:*),x1,phase,css
c
c
        x1 = x(1)
c
c
c       Get out of this routine if n = 1.
c
        if(n .eq. 1) then
          css = x1
          scal = 0
          return
        endif
c
c
c       Calculate |x(2)|^2 + ... |x(n)|^2
c       and the root-sum-square value of the entries in x.
c
c
        sum = 0
        do k = 2,n
          sum = sum+x(k)*conjg(x(k))
        enddo ! k
c
c
c       Get out of this routine if sum = 0;
c       flag this case as such by setting v(2), ..., v(n) all to 0.
c
        if(sum .eq. 0) then
c
          css = x1
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
        rss = x1*conjg(x1) + sum
        rss = sqrt(rss)
c
c
c       Determine the first component v1
c       of the unnormalized Householder vector
c       v = x - phase(x1) * rss * (1 0 0 ... 0 0)^T.
c
        if(x1 .eq. 0) phase = 1
        if(x1 .ne. 0) phase = x1/abs(x1)
        test = conjg(phase) * x1
        css = phase*rss
c
c       If test <= 0, then form x1-phase*rss directly,
c       since that expression cannot involve any cancellation.
c
        if(test .le. 0) v1 = x1-phase*rss
c
c       If test > 0, then use the fact that
c       x1-phase*rss = -phase*sum / ((phase)^* * x1 + rss),
c       in order to avoid potential cancellation.
c
        if(test .gt. 0) v1 = -phase*sum / (conjg(phase)*x1+rss)
c
c
c       Compute the vector vn and the scalar scal such that vn(1) = 1
c       in the Householder transformation
c       identity_matrix - scal * vn * adjoint(vn).
c
        do k = 2,n
          vn(k) = x(k)/v1
        enddo ! k
c
c       scal = 2
c            / ( |vn(1)|^2 + |vn(2)|^2 + ... + |vn(n)|^2 )
c
c            = 2
c            / ( 1 + |vn(2)|^2 + ... + |vn(n)|^2 )
c
c            = 2*|v(1)|^2
c            / ( |v(1)|^2 + |v(1)*vn(2)|^2 + ... + |v(1)*vn(n)|^2 )
c
c            = 2*|v(1)|^2
c            / ( |v(1)|^2 + (|v(2)|^2 + ... + |v(n)|^2) )
c
        scal = 2*v1*conjg(v1) / (v1*conjg(v1)+sum)
c
c
        rss = phase*rss
c
c
        return
        end
c
c
c
c
