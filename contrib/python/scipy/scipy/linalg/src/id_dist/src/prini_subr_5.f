        subroutine mach_zero(zero_mach)
        implicit real *8 (a-h,o-z)
        save
c
        zero_mach=100       
c
        d1=1.1
        d3=1.1
        d=1.11
        do 1200 i=1,1000
c

        d=d/2
        d2=d1+d
        call mach_zero0(d2,d3,d4)
c
        if(d4 .eq. 0) goto 1400
c
 1200 continue
 1400 continue
c
        zero_mach=d
        return
        end

c 
c 
c 
c 
c 
