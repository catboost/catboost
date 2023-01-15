        subroutine iddr_aidi(m,n,krank,w)
c
c       initializes the array w for using routine iddr_aid.
c
c       input:
c       m -- number of rows in the matrix to be ID'd
c       n -- number of columns in the matrix to be ID'd
c       krank -- rank of the ID to be constructed
c
c       output:
c       w -- initialization array for using routine iddr_aid
c
        implicit none
        integer m,n,krank,l,n2
        real*8 w((2*krank+17)*n+27*m+100)
c
c
c       Set the number of random test vectors to 8 more than the rank.
c
        l = krank+8
        w(1) = l
c
c
c       Initialize the rest of the array w.
c
        n2 = 0
        if(l .le. m) call idd_sfrmi(l,m,n2,w(11))
        w(2) = n2
c
c
        return
        end
