        subroutine iddr_svd(m,n,a,krank,u,v,s,ier,r)
c
c       constructs a rank-krank SVD  u diag(s) v^T  approximating a,
c       where u is an m x krank matrix whose columns are orthonormal,
c       v is an n x krank matrix whose columns are orthonormal,
c       and diag(s) is a diagonal krank x krank matrix whose entries
c       are all nonnegative. This routine combines a QR code
c       (which is based on plane/Householder reflections)
c       with the LAPACK routine dgesdd.
c
c       input:
c       m -- first dimension of a and u
c       n -- second dimension of a, and first dimension of v
c       a -- matrix to be SVD'd
c       krank -- desired rank of the approximation to a
c
c       output:
c       u -- left singular vectors of a corresponding
c            to the k greatest singular values of a
c       v -- right singular vectors of a corresponding
c            to the k greatest singular values of a
c       s -- k greatest singular values of a
c       ier -- 0 when the routine terminates successfully;
c              nonzero when the routine encounters an error
c
c       work:
c       r -- must be at least
c            (krank+2)*n+8*min(m,n)+15*krank**2+8*krank
c            real*8 elements long
c
c       _N.B._: This routine destroys a. Also, please beware that
c               the source code for this routine could be clearer.
c
        implicit none
        character*1 jobz
        integer m,n,k,krank,iftranspose,ldr,ldu,ldvt,lwork,
     1          info,j,ier,io
        real*8 a(m,n),u(m,krank),v(n*krank),s(krank),r(*)
c
c
        io = 8*min(m,n)
c
c
        ier = 0
c
c
c       Compute a pivoted QR decomposition of a.
c
        call iddr_qrpiv(m,n,a,krank,r,r(io+1))
c
c
c       Extract R from the QR decomposition.
c
        call idd_retriever(m,n,a,krank,r(io+1))
c
c
c       Rearrange R according to ind (which is stored in r).
c
        call idd_permuter(krank,r,krank,n,r(io+1))
c
c
c       Use LAPACK to SVD R,
c       storing the krank (krank x 1) left singular vectors
c       in r(io+krank*n+1 : io+krank*n+krank*krank).
c
        jobz = 'S'
        ldr = krank
        lwork = 2*(3*krank**2+n+4*krank**2+4*krank)
        ldu = krank
        ldvt = krank
c
        call dgesdd(jobz,krank,n,r(io+1),ldr,s,r(io+krank*n+1),ldu,
     1              v,ldvt,r(io+krank*n+krank*krank+1),lwork,r,info)
c
        if(info .ne. 0) then
          ier = info
          return
        endif
c
c
c       Multiply the U from R from the left by Q to obtain the U
c       for A.
c
        do k = 1,krank
c
          do j = 1,krank
            u(j,k) = r(io+krank*n+j+krank*(k-1))
          enddo ! j
c
          do j = krank+1,m
            u(j,k) = 0
          enddo ! j
c
        enddo ! k
c
        iftranspose = 0
        call idd_qmatmat(iftranspose,m,n,a,krank,krank,u,r)
c
c
c       Transpose v to obtain r.
c
        call idd_transer(krank,n,v,r)
c
c
c       Copy r into v.
c
        do k = 1,n*krank
          v(k) = r(k)
        enddo ! k
c
c
        return
        end
c
c
c
c
