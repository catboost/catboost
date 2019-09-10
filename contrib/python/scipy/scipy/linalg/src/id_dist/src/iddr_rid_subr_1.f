        subroutine iddr_ridall0(m,n,matvect,p1,p2,p3,p4,krank,
     1                          list,r,x,y)
c
c       routine iddr_ridall serves as a memory wrapper
c       for the present routine
c       (see iddr_ridall for further documentation).
c
        implicit none
        integer j,k,l,m,n,krank,list(n)
        real*8 x(m),y(n),p1,p2,p3,p4,r(krank+2,n)
        external matvect
c
c
c       Set the number of random test vectors to 2 more than the rank.
c
        l = krank+2
c
c       Apply the transpose of the original matrix to l random vectors.
c
        do j = 1,l
c
c         Generate a random vector.
c
          call id_srand(m,x)
c
c         Apply the transpose of the matrix to x, obtaining y.
c
          call matvect(m,x,n,y,p1,p2,p3,p4)
c
c         Copy y into row j of r.
c
          do k = 1,n
            r(j,k) = y(k)
          enddo ! k
c
        enddo ! j
c
c
c       ID r.
c
        call iddr_id(l,n,r,krank,list,y)
c
c
        return
        end
