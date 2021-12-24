        subroutine idz_frm(m,n,w,x,y)
c
c       transforms x into y via a composition
c       of Rokhlin's random transform, random subselection, and an FFT.
c       In contrast to routine idz_sfrm, the present routine works best
c       when the length of the transformed vector is the integer n
c       output by routine idz_frmi, or when the length
c       is not specified, but instead determined a posteriori
c       using the output of the present routine. The transformed vector
c       output by the present routine is randomly permuted.
c
c       input:
c       m -- length of x
c       n -- greatest integer expressible as a positive integer power
c            of 2 that is less than or equal to m, as obtained
c            from the routine idz_frmi; n is the length of y
c       w -- initialization array constructed by routine idz_frmi
c       x -- vector to be transformed
c
c       output:
c       y -- transform of x
c
c       reference:
c       Halko, Martinsson, Tropp, "Finding structure with randomness:
c            probabilistic algorithms for constructing approximate
c            matrix decompositions," SIAM Review, 53 (2): 217-288,
c            2011.
c
        implicit none
        integer m,iw,n,k
        complex*16 w(17*m+70),x(m),y(n)
c
c
c       Apply Rokhlin's random transformation to x, obtaining
c       w(16*m+71 : 17*m+70).
c
        iw = w(3+m+n)
        call idz_random_transf(x,w(16*m+70+1),w(iw))
c
c
c       Subselect from  w(16*m+71 : 17*m+70)  to obtain y.
c
        call idz_subselect(n,w(3),m,w(16*m+70+1),y)
c
c
c       Copy y into  w(16*m+71 : 16*m+n+70).
c
        do k = 1,n
          w(16*m+70+k) = y(k)
        enddo ! k
c
c
c       Fourier transform  w(16*m+71 : 16*m+n+70).
c
        call zfftf(n,w(16*m+70+1),w(4+m+n))
c
c
c       Permute  w(16*m+71 : 16*m+n+70)  to obtain y.
c
        call idz_permute(n,w(3+m),w(16*m+70+1),y)
c
c
        return
        end
c
c
c
c
        subroutine idz_sfrm(l,m,n,w,x,y)
c
c       transforms x into y via a composition
c       of Rokhlin's random transform, random subselection, and an FFT.
c       In contrast to routine idz_frm, the present routine works best
c       when the length l of the transformed vector is known a priori.
c
c       input:
c       l -- length of y; l must be less than or equal to n
c       m -- length of x
c       n -- greatest integer expressible as a positive integer power
c            of 2 that is less than or equal to m, as obtained
c            from the routine idz_frmi
c       w -- initialization array constructed by routine idz_sfrmi
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
        integer m,iw,n,l
        complex*16 w(21*m+70),x(m),y(l)
c
c
c       Apply Rokhlin's random transformation to x, obtaining
c       w(19*m+71 : 20*m+70).
c
        iw = w(4+m+l)
        call idz_random_transf(x,w(19*m+70+1),w(iw))
c
c
c       Subselect from  w(19*m+71 : 20*m+70)  to obtain
c       w(20*m+71 : 20*m+n+70).
c
        call idz_subselect(n,w(4),m,w(19*m+70+1),w(20*m+70+1))
c
c
c       Fourier transform  w(20*m+71 : 20*m+n+70).
c
        call idz_sfft(l,w(4+m),n,w(5+m+l),w(20*m+70+1))
c
c
c       Copy the desired entries from  w(20*m+71 : 20*m+n+70)
c       to y.
c
        call idz_subselect(l,w(4+m),n,w(20*m+70+1),y)
c
c
        return
        end
c
c
c
c
