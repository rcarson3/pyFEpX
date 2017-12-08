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

SUBROUTINE invert4x4(m, v, invOut)
    !Took this from a stackoverflow topic on how to code a inverted 4x4 matrix
    IMPLICIT NONE

    REAL(RK), INTENT(IN)  :: m(0:15)
    REAL(RK), INTENT(OUT) :: invOut(0:15)
    REAL(RK) :: inv
    REAL(RK) :: det, idet

    inv(0) =  m(5) * m(10) * m(15) - m(5) * m(11) * m(14) - m(9)&
            &* m(6) * m(15) + m(9) * m(7) * m(14) + m(13) * m(6) * m(11) - m(13) * m(7) * m(10)

    inv(4) = -m(4) * m(10) * m(15) + m(4) * m(11) * m(14) + m(8)&
            &* m(6) * m(15) - m(8) * m(7) * m(14) - m(12) * m(6) * m(11) + m(12) * m(7) * m(10)

    inv(8) =  m(4) * m(9) * m(15) - m(4) * m(11) * m(13) - m(8)&
            &* m(5) * m(15) + m(8) * m(7) * m(13) + m(12) * m(5) * m(11) - m(12) * m(7) * m(9)

    inv(12) = -m(4) * m(9) * m(14) + m(4) * m(10) * m(13) + m(8)&
            &* m(5) * m(14) - m(8) * m(6) * m(13) - m(12) * m(5) * m(10) + m(12) * m(6) * m(9)

    inv(1) = -m(1) * m(10) * m(15) + m(1) * m(11) * m(14) + m(9)&
            &* m(2) * m(15) - m(9) * m(3) * m(14) - m(13) * m(2) * m(11) + m(13) * m(3) * m(10)

    inv(5) =  m(0) * m(10) * m(15) - m(0) * m(11) * m(14) - m(8)&
            &* m(2) * m(15) + m(8) * m(3) * m(14) + m(12) * m(2) * m(11) - m(12) * m(3) * m(10)

    inv(9) = -m(0) * m(9) * m(15) + m(0) * m(11) * m(13) + m(8)&
            &* m(1) * m(15) - m(8) * m(3) * m(13) - m(12) * m(1) * m(11) + m(12) * m(3) * m(9)

    inv(13) =  m(0) * m(9) * m(14) - m(0) * m(10) * m(13) - m(8)&
            &* m(1) * m(14) + m(8) * m(2) * m(13) + m(12) * m(1) * m(10) - m(12) * m(2) * m(9)

    inv(2) =  m(1) * m(6) * m(15) - m(1) * m(7) * m(14) - m(5)&
            &* m(2) * m(15) + m(5) * m(3) * m(14) + m(13) * m(2) * m(7) - m(13) * m(3) * m(6)

    inv(6) = -m(0) * m(6) * m(15) + m(0) * m(7) * m(14) + m(4)&
            &* m(2) * m(15) - m(4) * m(3) * m(14) - m(12) * m(2) * m(7) + m(12) * m(3) * m(6)

    inv(10) =  m(0) * m(5) * m(15) - m(0) * m(7) * m(13) - m(4)&
            &* m(1) * m(15) + m(4) * m(3) * m(13) + m(12) * m(1) * m(7) - m(12) * m(3) * m(5)

    inv(14) = -m(0) * m(5) * m(14) + m(0) * m(6) * m(13) + m(4)&
            &* m(1) * m(14) - m(4) * m(2) * m(13) - m(12) * m(1) * m(6) + m(12) * m(2) * m(5)

    inv(3) = -m(1) * m(6) * m(11) + m(1) * m(7) * m(10) + m(5)&
            &* m(2) * m(11) - m(5) * m(3) * m(10) - m(9) * m(2) * m(7) + m(9) * m(3) * m(6)

    inv(7) =  m(0) * m(6) * m(11) - m(0) * m(7) * m(10) - m(4)&
            &* m(2) * m(11) + m(4) * m(3) * m(10) + m(8) * m(2) * m(7) - m(8) * m(3) * m(6)

    inv(11) = -m(0) * m(5) * m(11) + m(0) * m(7) * m(9) + m(4)&
            &* m(1) * m(11) - m(4) * m(3) * m(9) - m(8) * m(1) * m(7) + m(8) * m(3) * m(5)

    inv(15) =  m(0) * m(5) * m(10) - m(0) * m(6) * m(9) - m(4)&
            &* m(1) * m(10) + m(4) * m(2) * m(9) + m(8) * m(1) * m(6) - m(8) * m(2) * m(5)

    det = m(0) * inv(0) + m(1) * inv(4) + m(2) * inv(8) + m(3) * inv(12);

    idet = 1.0_RK/det;

    invOut(:) = inv(:) * idet;

END SUBROUTINE invert4x4

SUBROUTINE solve4x4(m, v, x)

    !We are going to be using this for finding the barycentric coordinates of a tetrahedron
    !This can then be used to tell us if a point is inside a tetrahedron or not. It can also
    !be used to linearly interpolate scalar values out to the coords of the tetrahedron.

    IMPLICIT NONE

    REAL(RK), INTENT(IN)  :: m(4,4), v(4)
    REAL(RK), INTENT(OUT) :: x(4)

    REAL(RK) :: inv_m(4,4)
    !Just giving a rough parameter for machine precision
    REAL(RK), PARAMETER :: mach_eps = 2.2e-16

    call invert4x4(m, inv_m)

    x = matmul(inv_m, v)

    !If under machine precision we're just going to set the value to zero
    !This should also help us deal with points that lie on an edge or point of the
    !tetrahedron
    where(abs(x) .LE. mach_eps) x = 0.0_RK

END SUBROUTINE solve4x4

end module matrixMath