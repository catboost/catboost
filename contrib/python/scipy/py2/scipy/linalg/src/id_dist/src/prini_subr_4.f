        subroutine fileflush(iw)
        implicit real *8 (a-h,o-z)
c 
        save
        close(iw)
        open(iw,status='old')
        do 1400 i=1,1000000
c 
        read(iw,1200,end=1600)
 1200 format(1a1)
 1400 continue
 1600 continue
c 
        return
        end
  
  
c 
c 
c 
c 
c 
