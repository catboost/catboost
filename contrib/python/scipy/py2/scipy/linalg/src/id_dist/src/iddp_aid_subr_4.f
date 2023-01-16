        subroutine idd_estrank0(eps,m,n,a,w,n2,krank,ra,rat,scal)
c
c       routine idd_estrank serves as a memory wrapper
c       for the present routine. (Please see routine idd_estrank
c       for further documentation.)
c
        implicit none
        integer m,n,n2,krank,ifrescal,k,nulls,j
        real*8 a(m,n),ra(n2,n),scal(n2+1),eps,residual,
     1         w(17*m+70),rat(n,n2+1),ss,ssmax
c
c
c       Apply the random matrix to every column of a, obtaining ra.
c
        do k = 1,n
          call idd_frm(m,n2,w,a(1,k),ra(1,k))
        enddo ! k
c
c
c       Compute the sum of squares of the entries in each column of ra
c       and the maximum of all such sums.
c
        ssmax = 0
c
        do k = 1,n
c
          ss = 0
          do j = 1,m
            ss = ss+a(j,k)**2
          enddo ! j
c
          if(ss .gt. ssmax) ssmax = ss
c
        enddo ! k
c
c
c       Transpose ra to obtain rat.
c
        call idd_atransposer(n2,n,ra,rat)
c
c
        krank = 0
        nulls = 0
c
c
c       Loop until nulls = 7, krank+nulls = n2, or krank+nulls = n.
c
 1000   continue
c
c
          if(krank .gt. 0) then
c
c           Apply the previous Householder transformations
c           to rat(:,krank+1).
c
            ifrescal = 0
c
            do k = 1,krank
              call idd_houseapp(n-k+1,rat(1,k),rat(k,krank+1),
     1                          ifrescal,scal(k),rat(k,krank+1))
            enddo ! k
c
          endif ! krank .gt. 0
c
c
c         Compute the Householder vector associated
c         with rat(krank+1:*,krank+1).
c
          call idd_house(n-krank,rat(krank+1,krank+1),
     1                   residual,rat(1,krank+1),scal(krank+1))
          residual = abs(residual)
c
c
          krank = krank+1
          if(residual .le. eps*sqrt(ssmax)) nulls = nulls+1
c
c
        if(nulls .lt. 7 .and. krank+nulls .lt. n2
     1   .and. krank+nulls .lt. n)
     2   goto 1000
c
c
        if(nulls .lt. 7) krank = 0
c
c
        return
        end
c
c
c
c
