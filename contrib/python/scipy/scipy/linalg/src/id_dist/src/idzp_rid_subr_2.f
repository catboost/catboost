        subroutine idz_findrank0(lra,eps,m,n,matveca,p1,p2,p3,p4,
     1                           krank,ra,ier,x,y,scal)
c
c       routine idz_findrank serves as a memory wrapper
c       for the present routine. (Please see routine idz_findrank
c       for further documentation.)
c
        implicit none
        integer m,n,krank,ifrescal,k,lra,ier,m2
        real*8 eps,enorm
        complex*16 x(m),ra(n,2,*),p1,p2,p3,p4,scal(n+1),y(n),residual
        external matveca
c
c
        ier = 0
c
c
        krank = 0
c
c
c       Loop until the relative residual is greater than eps,
c       or krank = m or krank = n.
c
 1000   continue
c
c
          if(lra .lt. n*2*(krank+1)) then
            ier = -1000
            return
          endif
c
c
c         Apply the adjoint of a to a random vector.
c
          m2 = m*2
          call id_srand(m2,x)
          call matveca(m,x,n,ra(1,1,krank+1),p1,p2,p3,p4)
c
          do k = 1,n
            y(k) = ra(k,1,krank+1)
          enddo ! k
c
c
          if(krank .eq. 0) then
c
c           Compute the Euclidean norm of y.
c
            enorm = 0
c
            do k = 1,n
              enorm = enorm + y(k)*conjg(y(k))
            enddo ! k
c
            enorm = sqrt(enorm)
c
          endif ! krank .eq. 0
c
c
          if(krank .gt. 0) then
c
c           Apply the previous Householder transformations to y.
c
            ifrescal = 0
c
            do k = 1,krank
              call idz_houseapp(n-k+1,ra(1,2,k),y(k),
     1                          ifrescal,scal(k),y(k))
            enddo ! k
c
          endif ! krank .gt. 0
c
c
c         Compute the Householder vector associated with y.
c
          call idz_house(n-krank,y(krank+1),
     1                   residual,ra(1,2,krank+1),scal(krank+1))
c
c
          krank = krank+1
c
c
        if(abs(residual) .gt. eps*enorm
     1   .and. krank .lt. m .and. krank .lt. n)
     2   goto 1000
c
c
c       Delete the Householder vectors from the array ra.
c
        call idz_crunch(n,krank,ra)
c
c
        return
        end
c
c
c
c
