module matrixMath

USE IntrinsicTypesModule, RK=>REAL_KIND
USE units_mod

IMPLICIT NONE

CONTAINS

SUBROUTINE mat_mult_const_mat(a, b, c)

    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: a(:, :, :), b(:, :)
    REAL(RK), INTENT(OUT):: c(:, :, :)

    INTEGER :: i, j, k

    INTEGER lda, ldb, ldc, lta, ltb, n
!
!----------------------------------------------------------------------

    lda = ubound(a, 2)
    ldb = ubound(b, 1)
    lta = ubound(a, 3)
    ltb = ubound(b, 2)

    c = 0.0_RK

    do k = 1,ltb
      do j = 1,ldb
         do i = 1,lda
            c(:, i, k) = c(:, i, k) + a(:, i, j)*b(j, k)
         end do
      end do
    end do

END SUBROUTINE mat_mult_const_mat

SUBROUTINE mat_const_mult_mat(a, b, c)

    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: a(:, :), b(:, :, :)
    REAL(RK), INTENT(OUT):: c(:, :, :)

    INTEGER :: i, j, k

    INTEGER lda, ldb, lta, ltb
!
!----------------------------------------------------------------------

    lda = ubound(a, 1)
    ldb = ubound(b, 2)
    lta = ubound(a, 2)
    ltb = ubound(b, 3)

    c = 0.0_RK

    do k = 1,ltb
      do j = 1,ldb
         do i = 1,lda
            c(:, i, k) = c(:, i, k) + a(i, j)*b(:, j, k)
         end do
      end do
    end do

END SUBROUTINE mat_const_mult_mat

SUBROUTINE matrix_mult(a, b, c)


    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: a(:, :, :), b(:, :, :)
    REAL(RK), INTENT(OUT):: c(:, :, :)

    INTEGER :: i, j, k

    INTEGER lda, ldb, lta, ltb
!
!----------------------------------------------------------------------

    lda = ubound(a, 2)
    ldb = ubound(b, 2)
    lta = ubound(a, 3)
    ltb = ubound(b, 3)

    c = 0.0_RK

    do k = 1,ltb
      do j = 1,ldb
         do i = 1,lda
            c(:, i, k) = c(:, i, k) + a(:, i, j)*b(:, j, k)
         end do
      end do
    end do

END SUBROUTINE matrix_mult

SUBROUTINE mat_const_mult_vec(a, b, c)

    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: a(:, :), b(:, :)
    REAL(RK), INTENT(OUT):: c(:, :)

    INTEGER :: j, k

    INTEGER lda, ldb , lta, ltb
!
!----------------------------------------------------------------------

    lda = ubound(a, 1)
    ldb = ubound(b, 1)
    lta = ubound(a, 2)
    ltb = ubound(b, 2)

    c = 0.0_RK

    do k = 1,ltb
      do j = 1,lda
        c(:, j) = c(:, j) + a(j, k)*b(:, k)
      end do
    end do

END SUBROUTINE mat_const_mult_vec

SUBROUTINE vec_dot_prod(a, b, c)

    IMPLICIT NONE
    REAL(RK), INTENT(IN) :: a(:, :), b(:, :)
    REAL(RK), INTENT(OUT) :: c(:)



    INTEGER :: i, lta
    lta = ubound(a, 2)

    c = 0.0_RK

    do i = 1,lta
        c(:) = c(:) + a(:, i)*b(:, i)
    end do

END SUBROUTINE vec_dot_prod

SUBROUTINE ltwonorm(vec, norm)

    IMPLICIT NONE
    REAL(RK), INTENT(IN) :: vec(:)
    REAL(RK), INTENT(OUT) :: norm

    REAL(RK), ALLOCATABLE :: svec(:)
    INTEGER :: i, lta

    lta = ubound(vec,1)
    ALLOCATE(svec(1:lta))

    svec = vec*vec
    norm = 0.0_RK

    do i = 1, lta
        norm = norm + sqrt(svec(i))
    end do

END SUBROUTINE ltwonorm

SUBROUTINE getskewmat(mat, skew, dim)

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: dim
    REAL(RK), INTENT(IN) :: mat(0:dim, 0:DIMS1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: skew(0:dim, 0:DIMS1, 0:DIMS1)

    skew = 0.0_RK

    skew(:, 0, 1) = 0.5_RK * (mat(:, 0, 1) - mat(:, 1, 0))
    skew(:, 0, 2) = 0.5_RK * (mat(:, 0, 2) - mat(:, 2, 0))
    skew(:, 1, 2) = 0.5_RK * (mat(:, 1, 2) - mat(:, 2, 1))

    skew(:, 1, 0) = - skew(:, 0, 1)
    skew(:, 2, 0) = - skew(:, 0, 2)
    skew(:, 2, 1) = - skew(:, 1, 2)


END SUBROUTINE getskewmat

SUBROUTINE getsymmat(mat, sym, dim)

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: dim
    REAL(RK), INTENT(IN) :: mat(0:dim, 0:DIMS1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: sym(0:dim, 0:DIMS1, 0:DIMS1)

    sym = 0.0_RK

    sym(:, 0, 0) = mat(:, 0, 0)
    sym(:, 1, 1) = mat(:, 1, 1)
    sym(:, 2, 2) = mat(:, 2, 2)

    sym(:, 0, 1) = 0.5_RK * (mat(:, 0, 1) + mat(:, 1, 0))
    sym(:, 0, 2) = 0.5_RK * (mat(:, 0, 2) + mat(:, 2, 0))
    sym(:, 1, 2) = 0.5_RK * (mat(:, 1, 2) + mat(:, 2, 1))

    sym(:, 1, 0) = sym(:, 0, 1)
    sym(:, 2, 0) = sym(:, 0, 2)
    sym(:, 2, 1) = sym(:, 1, 2)


END SUBROUTINE getsymmat

SUBROUTINE getDevMat(mat, dev)

    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: mat(0:nelem1, 0:DIMS1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: dev(0:nelem1, 0:DIMS1, 0:DIMS1)

    REAL(RK) :: trVal(0:nelem1)

    trVal = 1.0_RK/3.0_RK * (mat(:, 0, 0)+mat(:, 1, 1)+mat(:, 2, 2))

    dev(:, 0, 0) = mat(:, 0, 0) - trVal
    dev(:, 1, 1) = mat(:, 1, 1) - trVal
    dev(:, 2, 2) = mat(:, 2, 2) - trVal

    dev(:, 0, 1) = mat(:, 0, 1)
    dev(:, 0, 2) = mat(:, 0, 2)
    dev(:, 1, 2) = mat(:, 1, 2)

    dev(:, 1, 0) = mat(:, 1, 0)
    dev(:, 2, 0) = mat(:, 2, 0)
    dev(:, 2, 1) = mat(:, 2, 1)


END SUBROUTINE getDevMat

end module matrixMath