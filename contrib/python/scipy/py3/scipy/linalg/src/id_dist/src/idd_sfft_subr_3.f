        subroutine idd_sfft1(ind,n,v,wsave)
c
c       routine idd_sfft serves as a wrapper around
c       the present routine; please see routine idd_sfft
c       for documentation.
c
        implicit none
        integer ind,n,k
        real*8 v(n),r1,twopi,sumr,sumi,fact,wsave(2*(2+15+4*n))
c
        r1 = 1
        twopi = 2*4*atan(r1)
c
c
        if(ind .lt. n/2) then
c
c
          sumr = 0
c
          do k = 1,n
            sumr = sumr+wsave(k)*v(k)
          enddo ! k
c
c
          sumi = 0
c
          do k = 1,n
            sumi = sumi+wsave(n+k)*v(k)
          enddo ! k
c
c
        endif ! ind .lt. n/2
c
c
        if(ind .eq. n/2) then
c
c
          fact = 1/sqrt(r1*n)
c
c
          sumr = 0
c
          do k = 1,n
            sumr = sumr+v(k)
          enddo ! k
c
          sumr = sumr*fact
c
c
          sumi = 0
c
          do k = 1,n/2
            sumi = sumi+v(2*k-1)
            sumi = sumi-v(2*k)
          enddo ! k
c
          sumi = sumi*fact
c
c
        endif ! ind .eq. n/2
c
c
        v(2*ind-1) = sumr
        v(2*ind) = sumi
c
c
        return
        end
c
c
c
c
