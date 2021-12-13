        subroutine idd_copyints(n,ia,ib)
c
c       copies ia into ib.
c
c       input:
c       n -- length of ia and ib
c       ia -- array to be copied
c
c       output:
c       ib -- copy of ia
c
        implicit none
        integer n,ia(n),ib(n),k
c
c
        do k = 1,n
          ib(k) = ia(k)
        enddo ! k
c
c
        return
        end
c
c
c
c
