        subroutine idd_random_transf_init(nsteps,n,w,keep)
        implicit real *8 (a-h,o-z)
        save
        dimension w(*)
c
c       prepares and stores in array w the data used
c       by the routines idd_random_transf and idd_random_transf_inverse
c       to apply rapidly a random orthogonal matrix
c       to an arbitrary user-specified vector.
c
c       input:
c       nsteps -- the degree of randomness of the operator
c                 to be applied
c       n -- the size of the matrix to be applied
c
c       output:
c       w -- the first keep elements of w contain all the data
c            to be used by routines idd_random_tranf
c            and idd_random_transf_inverse. Please note that
c            the number of elements used by the present routine
c            is also equal to keep. This array should be at least
c            3*nsteps*n + 2*n + n/4 + 50 real*8 elements long.
c       keep - the number of elements in w actually used
c              by the present routine; keep is also the number
c              of elements that must not be changed between the call
c              to this routine and subsequent calls to routines
c              idd_random_transf and idd_random_transf_inverse.
c
c
c        . . . allocate memory
c
        ninire=2
c
        ialbetas=10
        lalbetas=2*n*nsteps+10
c
        iixs=ialbetas+lalbetas
        lixs=n*nsteps/ninire+10
c
        iww=iixs+lixs
        lww=2*n+n/4+20
c
        keep=iww+lww
c
        w(1)=ialbetas+0.1
        w(2)=iixs+0.1
        w(3)=nsteps+0.1
        w(4)=iww+0.1
        w(5)=n+0.1
c
        call idd_random_transf_init0(nsteps,n,w(ialbetas),w(iixs))
c
        return
        end
c
c
c
c
c
        subroutine idz_random_transf_init(nsteps,n,w,keep)
        implicit real *8 (a-h,o-z)
        save
        dimension w(*)
c
c       prepares and stores in array w the data used
c       by routines idz_random_transf and idz_random_transf_inverse
c       to apply rapidly a random unitary matrix
c       to an arbitrary user-specified vector.
c
c       input:
c       nsteps -- the degree of randomness of the operator
c                 to be applied
c       n -- the size of the matrix to be applied
c
c       output:
c       w -- the first keep elements of w contain all the data
c            to be used by routines idz_random_transf
c            and idz_random_transf_inverse. Please note that
c            the number of elements used by the present routine
c            is also equal to keep. This array should be at least
c            5*nsteps*n + 2*n + n/4 + 60 real*8 elements long.
c       keep - the number of elements in w actually used
c              by the present routine; keep is also the number
c              of elements that must not be changed between the call
c              to this routine and subsequent calls to routines
c              idz_random_transf and idz_random_transf_inverse.
c
c
c        . . . allocate memory
c
        ninire=2
c
        ialbetas=10
        lalbetas=2*n*nsteps+10
c
        igammas=ialbetas+lalbetas
        lgammas=2*n*nsteps+10
c
        iixs=igammas+lgammas
        lixs=n*nsteps/ninire+10
c
        iww=iixs+lixs
        lww=2*n+n/4+20
c
        keep=iww+lww
c
        w(1)=ialbetas+0.1
        w(2)=iixs+0.1
        w(3)=nsteps+0.1
        w(4)=iww+0.1
        w(5)=n+0.1
        w(6)=igammas+0.1
c
        call idz_random_transf_init0(nsteps,n,w(ialbetas),
     1      w(igammas),w(iixs))
c
        return
        end
c
c
c
c
c
        subroutine idd_random_transf(x,y,w)
        implicit real *8 (a-h,o-z)
        save
        dimension x(*),y(*),w(*)
c
c       applies rapidly a random orthogonal matrix
c       to the user-specified real vector x,
c       using the data in array w stored there by a preceding
c       call to routine idd_random_transf_init.
c
c       input:
c       x -- the vector of length n to which the random matrix is
c            to be applied
c       w -- array containing all initialization data
c
c       output:
c       y -- the result of applying the random matrix to x
c
c
c        . . . allocate memory
c
        ialbetas=w(1)
        iixs=w(2)
        nsteps=w(3)
        iww=w(4)
        n=w(5)
c
        call idd_random_transf0(nsteps,x,y,n,w(iww),
     1      w(ialbetas),w(iixs))
c
        return
        end
c
c
c
c
c
        subroutine idd_random_transf_inverse(x,y,w)
        implicit real *8 (a-h,o-z)
        save
        dimension x(*),y(*),w(*)
c
c       applies rapidly a random orthogonal matrix
c       to the user-specified real vector x,
c       using the data in array w stored there by a preceding
c       call to routine idd_random_transf_init.
c       The transformation applied by the present routine is
c       the inverse of the transformation applied
c       by routine idd_random_transf.
c
c       input:
c       x -- the vector of length n to which the random matrix is
c            to be applied
c       w -- array containing all initialization data
c
c       output:
c       y -- the result of applying the random matrix to x
c
c
c        . . . allocate memory
c
        ialbetas=w(1)
        iixs=w(2)
        nsteps=w(3)
        iww=w(4)
        n=w(5)
c
        call idd_random_transf0_inv(nsteps,x,y,n,w(iww),
     1      w(ialbetas),w(iixs))
c
        return
        end
c
c
c
c
c
        subroutine idz_random_transf(x,y,w)
        implicit real *8 (a-h,o-z)
        save
        complex *16 x(*),y(*)
        dimension w(*)
c
c       applies rapidly a random unitary matrix
c       to the user-specified vector x,
c       using the data in array w stored there by a preceding
c       call to routine idz_random_transf_init.
c
c       input:
c       x -- the vector of length n to which the random matrix is
c            to be applied
c       w -- array containing all initialization data
c
c       output:
c       y -- the result of applying the random matrix to x
c
c
c        . . . allocate memory
c
        ialbetas=w(1)
        iixs=w(2)
        nsteps=w(3)
        iww=w(4)
        n=w(5)
        igammas=w(6)
c
        call idz_random_transf0(nsteps,x,y,n,w(iww),w(ialbetas),
     1      w(igammas),w(iixs))
c
        return
        end
c
c
c
c
c
        subroutine idz_random_transf_inverse(x,y,w)
        implicit real *8 (a-h,o-z)
        save
        complex *16 x(*),y(*)
        dimension w(*)
c
c       applies rapidly a random unitary matrix
c       to the user-specified vector x,
c       using the data in array w stored there by a preceding
c       call to routine idz_random_transf_init.
c       The transformation applied by the present routine is
c       the inverse of the transformation applied
c       by routine idz_random_transf.
c
c       input:
c       x -- the vector of length n to which the random matrix is
c            to be applied
c       w -- array containing all initialization data
c
c       output:
c       y -- the result of applying the random matrix to x
c
c
c        . . . allocate memory
c
        ialbetas=w(1)
        iixs=w(2)
        nsteps=w(3)
        iww=w(4)
        n=w(5)
        igammas=w(6)
c
        call idz_random_transf0_inv(nsteps,x,y,n,w(iww),
     1      w(ialbetas),w(igammas),w(iixs))
c
        return
        end
c
c
c
c
c
