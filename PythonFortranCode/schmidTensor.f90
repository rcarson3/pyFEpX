module schmidTensor

USE IntrinsicTypesModule, RK=>REAL_KIND
USE ConstantsModule
USE matrixMath

IMPLICIT NONE

PUBLIC :: schmid, pschmid, qschmid

REAL(RK), ALLOCATABLE :: schmid(:, :, :)

REAL(RK), ALLOCATABLE :: pschmid(:, :, :)
REAL(RK), ALLOCATABLE :: qschmid(:, :, :)

CONTAINS

SUBROUTINE setSchmidTensor()

    IMPLICIT NONE

    REAL(RK), PARAMETER :: Z  =  RK_ZERO
    REAL(RK), PARAMETER :: P2 =  RK_ONE/RK_ROOT_2, P3 = RK_ONE/RK_ROOT_3
    REAL(RK), PARAMETER :: M2 = -P2, M3 = -P3

    !
    !  SLIP NORMAL AND DIRECTIONS.
    !
    REAL(RK), PARAMETER, DIMENSION(36) :: cub_111_dat = (/&
        &   P3, P3, P3,     P3, P3, P3,    P3, P3, P3,&
        &   P3, P3, M3,     P3, P3, M3,    P3, P3, M3,&
        &   P3, M3, P3,     P3, M3, P3,    P3, M3, P3,&
        &   P3, M3, M3,     P3, M3, M3,    P3, M3, M3 &
        &  /)
    REAL(RK), PARAMETER, DIMENSION(3, 12) :: &
        &   cub_111 = RESHAPE(SOURCE=cub_111_dat, SHAPE=(/3, 12/))
    !
    REAL(RK), PARAMETER, DIMENSION(36) :: cub_110_dat = (/&
        &  Z, P2, M2,    P2, Z, M2,    P2, M2, Z,&
        &  Z, P2, P2,    P2, Z, P2,    P2, M2, Z,&
        &  Z, P2, P2,    P2, Z, M2,    P2, P2, Z,&
        &  Z, P2, M2,    P2, Z, P2,    P2, P2, Z &
        &  /)
    REAL(RK), PARAMETER, DIMENSION(3, 12) :: &
        &   cub_110 = RESHAPE(SOURCE=cub_110_dat, SHAPE=(/3, 12/))

    !FCC Crystal only
    ALLOCATE(SCHMID(0:11, 0:2, 0:2),PSCHMID(0:11, 0:2, 0:2),QSCHMID(0:11, 0:2, 0:2))
    !
    schmid(:, 0, 0) = cub_110(1, :)*cub_111(1, :)
    schmid(:, 1, 0) = cub_110(2, :)*cub_111(1, :)
    schmid(:, 2, 0) = cub_110(3, :)*cub_111(1, :)

    schmid(:, 0, 1) = cub_110(1, :)*cub_111(2, :)
    schmid(:, 1, 1) = cub_110(2, :)*cub_111(2, :)
    schmid(:, 2, 1) = cub_110(3, :)*cub_111(2, :)

    schmid(:, 0, 2) = cub_110(1, :)*cub_111(3, :)
    schmid(:, 1, 2) = cub_110(2, :)*cub_111(3, :)
    schmid(:, 2, 2) = cub_110(3, :)*cub_111(3, :)

    call getSymMat(schmid, pschmid, 11)

    call getskewMat(schmid, qschmid, 11)


END SUBROUTINE setSchmidTensor

SUBROUTINE deallocate_schmid()

    IMPLICIT NONE

    DEALLOCATE(schmid, pschmid, qschmid)

END SUBROUTINE deallocate_schmid

end module schmidTensor
