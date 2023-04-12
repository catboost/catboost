        subroutine idd_id2svd0(m,krank,b,n,list,proj,u,v,s,ier,
     1                         work,p,t,r,r2,r3,ind,indt)
c
c       routine idd_id2svd serves as a memory wrapper
c       for the present routine (please see routine idd_id2svd
c       for further documentation).
c
        implicit none
c
        character*1 jobz
        integer m,n,krank,list(n),ind(n),indt(m),iftranspose,
     1          lwork,ldu,ldvt,ldr,info,j,k,ier
        real*8 b(m,krank),proj(krank,n-krank),p(krank,n),
     1         r(krank,n),r2(krank,m),t(n,krank),r3(krank,krank),
     2         u(m,krank),v(n,krank),s(krank),work(25*krank**2)
c
c
c
        ier = 0
c
c
c
c       Construct the projection matrix p from the ID.
c
        call idd_reconint(n,list,krank,proj,p)
c
c
c
c       Compute a pivoted QR decomposition of b.
c
        call iddr_qrpiv(m,krank,b,krank,ind,r)
c
c
c       Extract r from the QR decomposition.
c
        call idd_rinqr(m,krank,b,krank,r)
c
c
c       Rearrange r according to ind.
c
        call idd_rearr(krank,ind,krank,krank,r)
c
c
c
c       Transpose p to obtain t.
c
        call idd_mattrans(krank,n,p,t)
c
c
c       Compute a pivoted QR decomposition of t.
c
        call iddr_qrpiv(n,krank,t,krank,indt,r2)
c
c
c       Extract r2 from the QR decomposition.
c
        call idd_rinqr(n,krank,t,krank,r2)
c
c
c       Rearrange r2 according to indt.
c
        call idd_rearr(krank,indt,krank,krank,r2)
c
c
c
c       Multiply r and r2^T to obtain r3.
c
        call idd_matmultt(krank,krank,r,krank,r2,r3)
c
c
c
c       Use LAPACK to SVD r3.
c
        jobz = 'S'
        ldr = krank
        lwork = 25*krank**2-krank**2-4*krank
        ldu = krank
        ldvt = krank
c
        call dgesdd(jobz,krank,krank,r3,ldr,s,work,ldu,r,ldvt,
     1              work(krank**2+4*krank+1),lwork,
     2              work(krank**2+1),info)
c
        if(info .ne. 0) then
          ier = info
          return
        endif
c
c
c
c       Multiply the u from r3 from the left by the q from b
c       to obtain the u for a.
c
        do k = 1,krank
c
          do j = 1,krank
            u(j,k) = work(j+krank*(k-1))
          enddo ! j
c
          do j = krank+1,m
            u(j,k) = 0
          enddo ! j
c
        enddo ! k
c
        iftranspose = 0
        call idd_qmatmat(iftranspose,m,krank,b,krank,krank,u,r2)
c
c
c
c       Transpose r to obtain r2.
c
        call idd_mattrans(krank,krank,r,r2)
c
c
c       Multiply the v from r3 from the left by the q from p^T
c       to obtain the v for a.
c
        do k = 1,krank
c
          do j = 1,krank
            v(j,k) = r2(j,k)
          enddo ! j
c
          do j = krank+1,n
            v(j,k) = 0
          enddo ! j
c
        enddo ! k
c
        iftranspose = 0
        call idd_qmatmat(iftranspose,n,krank,t,krank,krank,v,r2)
c
c
        return
        end
c
c
c
c
