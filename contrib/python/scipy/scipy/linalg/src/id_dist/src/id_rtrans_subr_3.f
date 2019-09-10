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
