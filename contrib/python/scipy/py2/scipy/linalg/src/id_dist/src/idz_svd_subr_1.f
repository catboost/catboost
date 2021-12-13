        subroutine idzp_svd(lw,eps,m,n,a,krank,iu,iv,is,w,ier)
c
c       constructs a rank-krank SVD  U Sigma V^*  approximating a
c       to precision eps, where U is an m x krank matrix whose
c       columns are orthonormal, V is an n x krank matrix whose
c       columns are orthonormal, and Sigma is a diagonal krank x krank
c       matrix whose entries are all nonnegative.
c       The entries of U are stored in w, starting at w(iu);
c       the entries of V are stored in w, starting at w(iv).
c       The diagonal entries of Sigma are stored in w,
c       starting at w(is). This routine combines a QR code
c       (which is based on plane/Householder reflections)
c       with the LAPACK routine zgesdd.
c
c       input:
c       lw -- maximum usable length of w (in complex*16 elements)
c       eps -- precision to which the SVD approximates a
c       m -- first dimension of a and u
c       n -- second dimension of a, and first dimension of v
c       a -- matrix to be SVD'd
c
c       output:
c       krank -- rank of the approximation to a
c       iu -- index in w of the first entry of the matrix
c             of orthonormal left singular vectors of a
c       iv -- index in w of the first entry of the matrix
c             of orthonormal right singular vectors of a
c       is -- index in w of the first entry of the array
c             of singular values of a; the singular values are stored
c             as complex*16 numbers whose imaginary parts are zeros
c       w -- array containing the singular values and singular vectors
c            of a; w doubles as a work array, and so must be at least
c            (krank+1)*(m+2*n+9)+8*min(m,n)+6*krank**2
c            complex*16 elements long, where krank is the rank
c            output by the present routine
c       ier -- 0 when the routine terminates successfully;
c              -1000 when lw is too small;
c              other nonzero values when zgesdd bombs
c
c       _N.B._: This routine destroys a. Also, please beware that
c               the source code for this routine could be clearer.
c               w must be at least
c               (krank+1)*(m+2*n+9)+8*min(m,n)+6*krank**2
c               complex*16 elements long, where krank is the rank
c               output by the present routine.
c
        implicit none
        character*1 jobz
        integer m,n,k,krank,ifadjoint,ldr,ldu,ldvadj,lwork,
     1          info,j,ier,io,iu,iv,is,ivi,isi,lu,lv,ls,lw
        real*8 eps
        complex*16 a(m,n),w(*)
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
        call idzp_qrpiv(eps,m,n,a,krank,w,w(io+1))
c
c
        if(krank .gt. 0) then
c
c
c         Extract R from the QR decomposition.
c
          call idz_retriever(m,n,a,krank,w(io+1))
c
c
c         Rearrange R according to ind.
c
          call idz_permuter(krank,w,krank,n,w(io+1))
c
c
c         Use LAPACK to SVD R,
c         storing the krank (krank x 1) left singular vectors
c         in w(io+krank*n+1 : io+krank*n+krank*krank).
c
          jobz = 'S'
          ldr = krank
          lwork = 2*(krank**2+2*krank+n)
          ldu = krank
          ldvadj = krank
c
          ivi = io+krank*n+krank*krank+lwork+3*krank**2+4*krank+1
          lv = n*krank
c
          isi = ivi+lv
          ls = krank
c
          if(lw .lt. isi+ls+m*krank-1) then
            ier = -1000
            return
          endif
c
          call zgesdd(jobz,krank,n,w(io+1),ldr,w(isi),w(io+krank*n+1),
     1                ldu,w(ivi),ldvadj,w(io+krank*n+krank*krank+1),
     2                lwork,w(io+krank*n+krank*krank+lwork+1),w,info)
c
          if(info .ne. 0) then
            ier = info
            return
          endif
c
c
c         Take the adjoint of w(ivi:ivi+lv-1) to obtain V.
c
          iv = 1
          call idz_adjer(krank,n,w(ivi),w(iv))
c
c
c         Copy w(isi:isi+ls/2) into w(is:is+ls-1).
c
          is = iv+lv
c
          call idz_realcomp(ls,w(isi),w(is))
c
c
c         Multiply the U from R from the left by Q to obtain the U
c         for A.
c
          iu = is+ls
          lu = m*krank
c
          do k = 1,krank
c
            do j = 1,krank
              w(iu-1+j+krank*(k-1)) = w(io+krank*n+j+krank*(k-1))
            enddo ! j
c
          enddo ! k
c
          do k = krank,1,-1
c
            do j = m,krank+1,-1
              w(iu-1+j+m*(k-1)) = 0
            enddo ! j
c
            do j = krank,1,-1
              w(iu-1+j+m*(k-1)) = w(iu-1+j+krank*(k-1))
            enddo ! j
c
          enddo ! k
c
          ifadjoint = 0
          call idz_qmatmat(ifadjoint,m,n,a,krank,krank,w(iu),
     1                     w(iu+lu+1))
c
c
        endif ! krank .gt. 0
c
c
        return
        end
c
c
c
c
