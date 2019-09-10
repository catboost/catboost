        subroutine msgmerge(a,b,c)
        save
        character *1 a(1),b(1),c(1),ast
        data ast/'*'/
c 
        do 1200 i=1,1000
c 
        if(a(i) .eq. ast) goto 1400
        c(i)=a(i)
        iadd=i
 1200 continue
c 
 1400 continue
c 
        do 1800 i=1,1000
c 
        c(iadd+i)=b(i)
        if(b(i) .eq. ast) return
 1800 continue
        return
        end
c 
c 
c 
c 
c 
  
