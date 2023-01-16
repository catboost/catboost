        subroutine iddr_aid0(m,n,a,krank,w,list,proj,r)
c
c       routine iddr_aid serves as a memory wrapper
c       for the present routine
c       (see iddr_aid for further documentation).
c
        implicit none
        integer k,l,m,n2,n,krank,list(n),mn,lproj
        real*8 a(m,n),r(krank+8,2*n),proj(krank,n-krank),
     1         w(27*m+100+n)
c
c       Please note that the second dimension of r is 2*n
c       (instead of n) so that if krank+8 >= m/2, then
c       we can copy the whole of a into r.
c
c
c       Retrieve the number of random test vectors
c       and the greatest integer less than m that is
c       a positive integer power of two.
c
        l = w(1)
        n2 = w(2)
c
c
        if(l .lt. n2 .and. l .le. m) then
c
c         Apply the random matrix.
c
          do k = 1,n
            call idd_sfrm(l,m,n2,w(11),a(1,k),r(1,k))
          enddo ! k
c
c         ID r.
c
          call iddr_id(l,n,r,krank,list,w(26*m+101))
c
c         Retrieve proj from r.
c
          lproj = krank*(n-krank)
          call iddr_copydarr(lproj,r,proj)
c
        endif
c
c
        if(l .ge. n2 .or. l .gt. m) then
c
c         ID a directly.
c
          mn = m*n
          call iddr_copydarr(mn,a,r)
          call iddr_id(m,n,r,krank,list,w(26*m+101))
c
c         Retrieve proj from r.
c
          lproj = krank*(n-krank)
          call iddr_copydarr(lproj,r,proj)
c
        endif
c
c
        return
        end
c
c
c
c
