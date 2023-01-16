        subroutine idd_sfrm(l,m,n,w,x,y)
c
c       transforms x into y via a composition
c       of Rokhlin's random transform, random subselection, and an FFT.
c       In contrast to routine idd_frm, the present routine works best
c       when the length l of the transformed vector is known a priori.
c
c       input:
c       l -- length of y; l must be less than or equal to n
c       m -- length of x
c       n -- greatest integer expressible as a positive integer power
c            of 2 that is less than or equal to m, as obtained
c            from the routine idd_sfrmi
c       w -- initialization array constructed by routine idd_sfrmi
c       x -- vector to be transformed
c
c       output:
c       y -- transform of x
c
c       _N.B._: l must be less than or equal to n.
c
c       reference:
c       Halko, Martinsson, Tropp, "Finding structure with randomness:
c            probabilistic algorithms for constructing approximate
c            matrix decompositions," SIAM Review, 53 (2): 217-288,
c            2011.
c
        implicit none
        integer m,iw,n,l,l2
        real*8 w(27*m+90),x(m),y(l)
c
c
c       Retrieve the number of pairs of outputs to be calculated
c       via sfft.
c
        l2 = w(3)
c
c
c       Apply Rokhlin's random transformation to x, obtaining
c       w(25*m+91 : 26*m+90).
c
        iw = w(4+m+l+l2)
        call idd_random_transf(x,w(25*m+90+1),w(iw))
c
c
c       Subselect from  w(25*m+91 : 26*m+90)  to obtain
c       w(26*m+91 : 26*m+n+90).
c
        call idd_subselect(n,w(4),m,w(25*m+90+1),w(26*m+90+1))
c
c
c       Fourier transform  w(26*m+91 : 26*m+n+90).
c
        call idd_sfft(l2,w(4+m+l),n,w(5+m+l+l2),w(26*m+90+1))
c
c
c       Copy the desired entries from  w(26*m+91 : 26*m+n+90)
c       to y.
c
        call idd_subselect(l,w(4+m),n,w(26*m+90+1),y)
c
c
        return
        end
c
c
c
c
