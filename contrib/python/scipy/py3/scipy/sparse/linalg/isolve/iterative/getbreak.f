* -*- fortran -*-
*     GetBreak
      real 
     $     FUNCTION sGETBREAK()
*
*     Get breakdown parameter tolerance; for the test routine,
*     set to machine precision.
*
      IMPLICIT NONE
      real EPS, sLAMCH
*
      EPS = sLAMCH('EPS')
      sGETBREAK = EPS**2
*
      RETURN
*
      END
*     END FUNCTION sGETBREAK





      double precision 
     $     FUNCTION dGETBREAK()
*
*     Get breakdown parameter tolerance; for the test routine,
*     set to machine precision.
*
      IMPLICIT NONE
      double precision EPS, dLAMCH
*
      EPS = dLAMCH('EPS')
      dGETBREAK = EPS**2
*
      RETURN
*
      END
*     END FUNCTION dGETBREAK





      real 
     $     FUNCTION cGETBREAK()
*
*     Get breakdown parameter tolerance; for the test routine,
*     set to machine precision.
*
      IMPLICIT NONE
      real EPS, sLAMCH
*
      EPS = sLAMCH('EPS')
      cGETBREAK = EPS**2
*
      RETURN
*
      END
*     END FUNCTION cGETBREAK





      double precision 
     $     FUNCTION zGETBREAK()
*
*     Get breakdown parameter tolerance; for the test routine,
*     set to machine precision.
*
      IMPLICIT NONE
      double precision EPS, dLAMCH
*
      EPS = dLAMCH('EPS')
      zGETBREAK = EPS**2
*
      RETURN
*
      END
*     END FUNCTION zGETBREAK





