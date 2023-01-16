        subroutine idd_frmi(m,n,w)
c
c       initializes data for the routine idd_frm.
c
c       input:
c       m -- length of the vector to be transformed
c
c       output:
c       n -- greatest integer expressible as a positive integer power
c            of 2 that is less than or equal to m
c       w -- initialization array to be used by routine idd_frm
c
c
c       glossary for the fully initialized w:
c
c       w(1) = m
c       w(2) = n
c       w(3:2+m) stores a permutation of m objects
c       w(3+m:2+m+n) stores a permutation of n objects
c       w(3+m+n) = address in w of the initialization array
c                  for idd_random_transf
c       w(4+m+n:int(w(3+m+n))-1) stores the initialization array
c                                for dfft
c       w(int(w(3+m+n)):16*m+70) stores the initialization array
c                                for idd_random_transf
c
c
c       _N.B._: n is an output of the present routine;
c               this routine changes n.
c
c
        implicit none
        integer m,n,l,nsteps,keep,lw,ia
        real*8 w(17*m+70)
c
c
c       Find the greatest integer less than or equal to m
c       which is a power of two.
c
        call idd_poweroftwo(m,l,n)
c
c
c       Store m and n in w.
c
        w(1) = m
        w(2) = n
c
c
c       Store random permutations of m and n objects in w.
c
        call id_randperm(m,w(3))
        call id_randperm(n,w(3+m))
c
c
c       Store the address within w of the idd_random_transf_init
c       initialization data.
c
        ia = 4+m+n+2*n+15
        w(3+m+n) = ia
c
c
c       Store the initialization data for dfft in w.
c
        call dffti(n,w(4+m+n))
c
c
c       Store the initialization data for idd_random_transf_init in w.
c
        nsteps = 3
        call idd_random_transf_init(nsteps,m,w(ia),keep)
c
c
c       Calculate the total number of elements used in w.
c
        lw = 3+m+n+2*n+15 + 3*nsteps*m+2*m+m/4+50
c
        if(16*m+70 .lt. lw) then
          call prinf('lw = *',lw,1)
          call prinf('16m+70 = *',16*m+70,1)
          stop
        endif
c
c
        return
        end
c
c
c
c
        subroutine idd_sfrmi(l,m,n,w)
c
c       initializes data for the routine idd_sfrm.
c
c       input:
c       l -- length of the transformed (output) vector
c       m -- length of the vector to be transformed
c
c       output:
c       n -- greatest integer expressible as a positive integer power
c            of 2 that is less than or equal to m
c       w -- initialization array to be used by routine idd_sfrm
c
c
c       glossary for the fully initialized w:
c
c       w(1) = m
c       w(2) = n
c       w(3) = l2
c       w(4:3+m) stores a permutation of m objects
c       w(4+m:3+m+l) stores the indices of the l outputs which idd_sfft
c                    calculates
c       w(4+m+l:3+m+l+l2) stores the indices of the l2 pairs of outputs
c                         which idd_sfft calculates
c       w(4+m+l+l2) = address in w of the initialization array
c                     for idd_random_transf
c       w(5+m+l+l2:int(w(4+m+l+l2))-1) stores the initialization array
c                                      for idd_sfft
c       w(int(w(4+m+l+l2)):25*m+90) stores the initialization array
c                                   for idd_random_transf
c
c
c       _N.B._: n is an output of the present routine;
c               this routine changes n.
c
c
        implicit none
        integer l,m,n,idummy,nsteps,keep,lw,l2,ia
        real*8 w(27*m+90)
c
c
c       Find the greatest integer less than or equal to m
c       which is a power of two.
c
        call idd_poweroftwo(m,idummy,n)
c
c
c       Store m and n in w.
c
        w(1) = m
        w(2) = n
c
c
c       Store random permutations of m and n objects in w.
c
        call id_randperm(m,w(4))
        call id_randperm(n,w(4+m))
c
c
c       Find the pairs of integers covering the integers in
c       w(4+m : 3+m+(l+1)/2).
c
        call idd_pairsamps(n,l,w(4+m),l2,w(4+m+2*l),w(4+m+3*l))
        w(3) = l2
        call idd_copyints(l2,w(4+m+2*l),w(4+m+l))
c
c
c       Store the address within w of the idd_random_transf_init
c       initialization data.
c
        ia = 5+m+l+l2+4*l2+30+8*n
        w(4+m+l+l2) = ia
c
c
c       Store the initialization data for idd_sfft in w.
c
        call idd_sffti(l2,w(4+m+l),n,w(5+m+l+l2))
c
c
c       Store the initialization data for idd_random_transf_init in w.
c
        nsteps = 3
        call idd_random_transf_init(nsteps,m,w(ia),keep)
c
c
c       Calculate the total number of elements used in w.
c
        lw = 4+m+l+l2+4*l2+30+8*n + 3*nsteps*m+2*m+m/4+50
c
        if(25*m+90 .lt. lw) then
          call prinf('lw = *',lw,1)
          call prinf('25m+90 = *',25*m+90,1)
          stop
        endif
c
c
        return
        end
c
c
c
c
