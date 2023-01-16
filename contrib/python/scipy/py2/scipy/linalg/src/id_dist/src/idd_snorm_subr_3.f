        subroutine idd_diffsnorm0(m,n,matvect,p1t,p2t,p3t,p4t,
     1                            matvect2,p1t2,p2t2,p3t2,p4t2,
     2                            matvec,p1,p2,p3,p4,
     3                            matvec2,p12,p22,p32,p42,
     4                            its,snorm,u,u1,u2,v,v1,v2)
c
c       routine idd_diffsnorm serves as a memory wrapper
c       for the present routine. (Please see routine idd_diffsnorm
c       for further documentation.)
c
        implicit none
        integer m,n,its,it,k
        real*8 snorm,enorm,p1t,p2t,p3t,p4t,p1t2,p2t2,p3t2,p4t2,
     1         p1,p2,p3,p4,p12,p22,p32,p42,u(m),u1(m),u2(m),
     2         v(n),v1(n),v2(n)
        external matvect,matvec,matvect2,matvec2
c
c
c       Fill the real and imaginary parts of each entry
c       of the initial vector v with i.i.d. random variables
c       drawn uniformly from [-1,1].
c
        call id_srand(n,v)
c
        do k = 1,n
          v(k) = 2*v(k)-1
        enddo ! k
c
c
c       Normalize v.
c
        call idd_enorm(n,v,enorm)
c
        do k = 1,n
          v(k) = v(k)/enorm
        enddo ! k
c
c
        do it = 1,its
c
c         Apply a and a2 to v, obtaining u1 and u2.
c
          call matvec(n,v,m,u1,p1,p2,p3,p4)
          call matvec2(n,v,m,u2,p12,p22,p32,p42)
c
c         Form u = u1-u2.
c
          do k = 1,m
            u(k) = u1(k)-u2(k)
          enddo ! k
c
c         Apply a^T and (a2)^T to u, obtaining v1 and v2.
c
          call matvect(m,u,n,v1,p1t,p2t,p3t,p4t)
          call matvect2(m,u,n,v2,p1t2,p2t2,p3t2,p4t2)
c
c         Form v = v1-v2.
c
          do k = 1,n
            v(k) = v1(k)-v2(k)
          enddo ! k
c
c         Normalize v.
c
          call idd_enorm(n,v,snorm)
c
          if(snorm .gt. 0) then
c
            do k = 1,n
              v(k) = v(k)/snorm
            enddo ! k
c
          endif
c
          snorm = sqrt(snorm)
c
        enddo ! it
c
c
        return
        end
