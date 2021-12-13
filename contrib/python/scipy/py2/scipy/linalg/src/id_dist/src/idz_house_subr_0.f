        subroutine idz_houseapp(n,vn,u,ifrescal,scal,v)
c
c       applies the Householder matrix
c       identity_matrix - scal * vn * adjoint(vn)
c       to the vector u, yielding the vector v;
c
c       scal = 2/(1 + |vn(2)|^2 + ... + |vn(n)|^2)
c       when vn(2), ..., vn(n) don't all vanish;
c
c       scal = 0
c       when vn(2), ..., vn(n) do all vanish
c       (including when n = 1).
c
c       input:
c       n -- size of vn, u, and v, though the indexing on vn goes
c            from 2 to n
c       vn -- components 2 to n of the Householder vector vn;
c             vn(1) is assumed to be 1 
c       u -- vector to be transformed
c       ifrescal -- set to 1 to recompute scal from vn(2), ..., vn(n);
c                   set to 0 to use scal as input
c       scal -- see the entry for ifrescal in the decription
c               of the input
c
c       output:
c       scal -- see the entry for ifrescal in the decription
c               of the input
c       v -- result of applying the Householder matrix to u;
c            it's O.K. to have v be the same as u
c            in order to apply the matrix to the vector in place
c
c       reference:
c       Golub and Van Loan, "Matrix Computations," 3rd edition,
c            Johns Hopkins University Press, 1996, Chapter 5.
c
        implicit none
        save
        integer n,k,ifrescal
        real*8 scal,sum
        complex*16 vn(2:*),u(n),v(n),fact
c
c
c       Get out of this routine if n = 1.
c
        if(n .eq. 1) then
          v(1) = u(1)
          return
        endif
c
c
        if(ifrescal .eq. 1) then
c
c
c         Calculate |vn(2)|^2 + ... + |vn(n)|^2.
c
          sum = 0
          do k = 2,n
            sum = sum+vn(k)*conjg(vn(k))
          enddo ! k
c
c
c         Calculate scal.
c
          if(sum .eq. 0) scal = 0
          if(sum .ne. 0) scal = 2/(1+sum)
c
c
        endif
c
c
c       Calculate fact = scal * adjoint(vn) * u.
c
        fact = u(1)
c
        do k = 2,n
          fact = fact+conjg(vn(k))*u(k)
        enddo ! k
c
        fact = fact*scal
c
c
c       Subtract fact*vn from u, yielding v.
c      
        v(1) = u(1) - fact
c
        do k = 2,n
          v(k) = u(k) - fact*vn(k)
        enddo ! k
c
c
        return
        end
c
c
c
c
