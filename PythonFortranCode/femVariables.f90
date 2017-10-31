module femVariables

USE IntrinsicTypesModule, RK=>REAL_KIND
USE units_mod

IMPLICIT NONE

PUBLIC :: connect, coord, gforce, giforce, grot, grotnot, gstiff, i2, vel

REAL(RK), ALLOCATABLE :: coord(:,:), vel(:,:)
REAL(RK), ALLOCATABLE :: gforce(:), giforce(:), grot(:), grotnot(:), &
                        & gstiff(:,:,:)

INTEGER, ALLOCATABLE :: i2(:,:), connect(:,:)

CONTAINS

SUBROUTINE init_femvar()

    IMPLICIT NONE

    ALLOCATE(connect(0:nelem1, 0:NNPE), coord(0:ncrd1, 0:DIMS1), vel(0:ncrd1, 0:DIMS1))
    ALLOCATE(i2(0:nelem1, 0:KDIM1))
    ALLOCATE(gforce(0:ncvec1), giforce(0:ncvec1), grot(0:ncvec1), grotnot(0:ncvec1))
    ALLOCATE(gstiff(0:nelem1, 0:KDIM1, 0:KDIM1))

    coord = 0.0_RK
    vel = 0.0_RK
    i2 = 0
    connect = 0
    gforce = 0.0_RK
    giforce = 0.0_RK
    gstiff = 0.0_RK
    grot = 0.0_RK
    grotnot = 0.0_RK

END SUBROUTINE init_femvar

SUBROUTINE deallocate_femvar()

    IMPLICIT NONE

    DEALLOCATE(connect, coord, gforce, giforce, grot, grotnot, gstiff, i2, vel)

END SUBROUTINE deallocate_femvar

SUBROUTINE set_i2()

    IMPLICIT NONE

    INTEGER :: i

    do i = 0,nelem1

        i2(i, 0:2) = (/connect(i,0)*3, connect(i,0)*3+1, connect(i,0)*3+2 /)
        i2(i, 3:5) = (/connect(i,1)*3, connect(i,1)*3+1, connect(i,1)*3+2 /)
        i2(i, 6:8) = (/connect(i,2)*3, connect(i,2)*3+1, connect(i,2)*3+2 /)
        i2(i, 9:11) = (/connect(i,3)*3, connect(i,3)*3+1, connect(i,3)*3+2 /)
        i2(i, 12:14) = (/connect(i,4)*3, connect(i,4)*3+1, connect(i,4)*3+2 /)
        i2(i, 15:17) = (/connect(i,5)*3, connect(i,5)*3+1, connect(i,5)*3+2 /)
        i2(i, 18:20) = (/connect(i,6)*3, connect(i,6)*3+1, connect(i,6)*3+2 /)
        i2(i, 21:23) = (/connect(i,7)*3, connect(i,7)*3+1, connect(i,7)*3+2 /)
        i2(i, 24:26) = (/connect(i,8)*3, connect(i,8)*3+1, connect(i,8)*3+2 /)
        i2(i, 27:29) = (/connect(i,9)*3, connect(i,9)*3+1, connect(i,9)*3+2 /)

    end do

END SUBROUTINE set_i2

SUBROUTINE getElemVec (vec, matelem)

    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: vec(0:ncvec1)
    REAL(RK), INTENT(OUT) :: matelem(0:nelem1, 0:KDIM1)

    INTEGER :: i

    do i =0,NNPE

        matelem(:, i*3) = vec(i2(:, i*3))
        matelem(:, (i*3)+1) = vec(i2(:, (i*3)+1))
        matelem(:, (i*3)+2) = vec(i2(:, (i*3)+2))

    end do

END SUBROUTINE getElemVec

SUBROUTINE getElemCoord(crd, matelem)

    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: crd(0:ncrd1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: matelem(0:nelem1, 0:KDIM1)

    INTEGER :: i

    do i =0,NNPE

        matelem(:, (/i*3, i*3+1, i*3+2/)) = crd(connect(:,i), :)

    end do

END SUBROUTINE getElemCoord


end module femVariables