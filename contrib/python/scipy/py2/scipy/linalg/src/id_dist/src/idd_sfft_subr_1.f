        subroutine idd_sffti1(ind,n,wsave)
c
c       routine idd_sffti serves as a wrapper around
c       the present routine; please see routine idd_sffti
c       for documentation.
c
        implicit none
        integer ind,n,k
        real*8 r1,twopi,wsave(2*(2+15+4*n)),fact
c
        r1 = 1
        twopi = 2*4*atan(r1)
c
c
        fact = 1/sqrt(r1*n)
c
c
        do k = 1,n
          wsave(k) = cos(twopi*(k-1)*ind/(r1*n))*fact
        enddo ! k
c
        do k = 1,n
          wsave(n+k) = -sin(twopi*(k-1)*ind/(r1*n))*fact
        enddo ! k
c
c
        return
        end
c
c
c
c
