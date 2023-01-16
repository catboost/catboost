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
