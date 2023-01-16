        subroutine idd_qinqr(m,n,a,krank,q)
c
c       constructs the matrix q from iddp_qrpiv or iddr_qrpiv
c       (see the routine iddp_qrpiv or iddr_qrpiv
c       for more information).
c
c       input:
c       m -- first dimension of a; also, right now, q is m x m
c       n -- second dimension of a
c       a -- matrix output by iddp_qrpiv or iddr_qrpiv
c            (and denoted the same there)
c       krank -- numerical rank output by iddp_qrpiv or iddr_qrpiv
c                (and denoted the same there)
c
c       output:
c       q -- orthogonal matrix implicitly specified by the data in a
c            from iddp_qrpiv or iddr_qrpiv
c
c       Note:
c       Right now, this routine simply multiplies
c       one after another the krank Householder matrices
c       in the full QR decomposition of a,
c       in order to obtain the complete m x m Q factor in the QR.
c       This routine should instead use the following 
c       (more elaborate but more efficient) scheme
c       to construct a q dimensioned q(krank,m); this scheme
c       was introduced by Robert Schreiber and Charles Van Loan
c       in "A Storage-Efficient _WY_ Representation
c       for Products of Householder Transformations,"
c       _SIAM Journal on Scientific and Statistical Computing_,
c       Vol. 10, No. 1, pp. 53-57, January, 1989:
c
c       Theorem 1. Suppose that Q = _1_ + YTY^T is
c       an m x m orthogonal real matrix,
c       where Y is an m x k real matrix
c       and T is a k x k upper triangular real matrix.
c       Suppose also that P = _1_ - 2 v v^T is
c       a real Householder matrix and Q_+ = QP,
c       where v is an m x 1 real vector,
c       normalized so that v^T v = 1.
c       Then, Q_+ = _1_ + Y_+ T_+ Y_+^T,
c       where Y_+ = (Y v) is the m x (k+1) matrix
c       formed by adjoining v to the right of Y,
c                 ( T   z )
c       and T_+ = (       ) is
c                 ( 0  -2 )
c       the (k+1) x (k+1) upper triangular matrix
c       formed by adjoining z to the right of T
c       and the vector (0 ... 0 -2) with k zeroes below (T z),
c       where z = -2 T Y^T v.
c
c       Now, suppose that A is a (rank-deficient) matrix
c       whose complete QR decomposition has
c       the blockwise partioned form
c           ( Q_11 Q_12 ) ( R_11 R_12 )   ( Q_11 )
c       A = (           ) (           ) = (      ) (R_11 R_12).
c           ( Q_21 Q_22 ) (  0    0   )   ( Q_21 )
c       Then, the only blocks of the orthogonal factor
c       in the above QR decomposition of A that matter are
c                                                        ( Q_11 )
c       Q_11 and Q_21, _i.e._, only the block of columns (      )
c                                                        ( Q_21 )
c       interests us.
c       Suppose in addition that Q_11 is a k x k matrix,
c       Q_21 is an (m-k) x k matrix, and that
c       ( Q_11 Q_12 )
c       (           ) = _1_ + YTY^T, as in Theorem 1 above.
c       ( Q_21 Q_22 )
c       Then, Q_11 = _1_ + Y_1 T Y_1^T
c       and Q_21 = Y_2 T Y_1^T,
c       where Y_1 is the k x k matrix and Y_2 is the (m-k) x k matrix
c                   ( Y_1 )
c       so that Y = (     ).
c                   ( Y_2 )
c
c       So, you can calculate T and Y via the above recursions,
c       and then use these to compute the desired Q_11 and Q_21.
c
c
        implicit none
        integer m,n,krank,j,k,mm,ifrescal
        real*8 a(m,n),q(m,m),scal
c
c
c       Zero all of the entries of q.
c
        do k = 1,m
          do j = 1,m
            q(j,k) = 0
          enddo ! j
        enddo ! k
c
c
c       Place 1's along the diagonal of q.
c
        do k = 1,m
          q(k,k) = 1
        enddo ! k
c
c
c       Apply the krank Householder transformations stored in a.
c
        do k = krank,1,-1
          do j = k,m
            mm = m-k+1
            ifrescal = 1
            if(k .lt. m)
     1       call idd_houseapp(mm,a(k+1,k),q(k,j),ifrescal,scal,q(k,j))
          enddo ! j
        enddo ! k
c
c
        return
        end
c
c
c
c
