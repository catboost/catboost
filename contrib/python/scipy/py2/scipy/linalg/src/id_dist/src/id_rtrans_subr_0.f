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
