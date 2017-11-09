module mainLatProgram
USE IntrinsicTypesModule, RK=>REAL_KIND
IMPLICIT NONE

!REAL(RK), ALLOCATABLE :: rote(:)

CONTAINS

SUBROUTINE initializeAll(nel, ng, ncoord)

    USE lat_ori_fem_mod

    USE IntrinsicTypesModule, RK=>REAL_KIND

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: nel, ng, ncoord

    call initializedata(nel, ncoord, ng)

END SUBROUTINE initializeAll

SUBROUTINE setData(eps, gammadot, velocity, connectivity, crd, rot, nel1, dim1, ngd1, ncr1, nnp, ncvc1)

USE lat_ori_fem_mod
USE IntrinsicTypesModule, RK=>REAL_KIND

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: nel1, dim1, ngd1, ncr1, nnp, ncvc1

    REAL(RK), INTENT(IN) :: eps(0:nel1, 0:dim1, 0:dim1), gammadot(0:nel1, 0:ngd1)
    REAL(RK), INTENT(IN) :: velocity(0:ncr1, 0:dim1), rot(0:ncvc1), crd(0:ncr1, 0:dim1)
    INTEGER, INTENT(IN) :: connectivity(0:nel1, 0:nnp)

    call setDataUp(eps, gammadot, velocity, connectivity, crd, rot)

END SUBROUTINE setData

SUBROUTINE getData(ncvc1, rot)

USE lat_ori_fem_mod
USE IntrinsicTypesModule, RK=>REAL_KIND

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: ncvc1
    REAL(RK), INTENT(OUT) :: rot(0:ncvc1)

!    allocate(rote(0:ncvc1))

    rot = 0.0_RK

!    write(*,*) rot

    call returnData(rot)


!    rote = rot

END SUBROUTINE getData

SUBROUTINE getDefGrad(def_grad, det, disp, iqpt, nel1, DIM1, ncr1, cenval)

USE lat_ori_fem_mod
USE IntrinsicTypesModule, RK=>REAL_KIND

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: nel1
    INTEGER, INTENT(IN) :: DIM1
    INTEGER, INTENT(IN) :: ncr1

    REAL(RK), INTENT(OUT) :: def_grad(0:nel1, 0:DIM1, 0:DIM1)
    REAL(RK), INTENT(OUT) :: det(0:nel1) !the determinate for each element as change from local element and parent element
    INTEGER, INTENT(IN) :: iqpt ! if above true this pretty much ignored
    REAL(RK), INTENT(IN) :: disp(0:ncr1, 0:DIM1)
    INTEGER, INTENT(IN) :: cenval !Do you want centroid value

    if (cenval.eq.1) then

       call get_def_grad(def_grad, disp, det, .TRUE., iqpt, .TRUE.)
    
    else

       call get_def_grad(def_grad, disp, det, .TRUE., iqpt, .FALSE.)
    
    endif

END SUBROUTINE getDefGrad

SUBROUTINE getJacobianDet(det, nel1, nqp1)

    USE lat_ori_fem_mod
    USE IntrinsicTypesModule, RK=>REAL_KIND

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: nel1
    INTEGER, INTENT(IN) :: nqp1

    REAL(RK), INTENT(OUT) :: det(0:nel1, 0:nqp1) !the determinate for each element as change from local element and parent element

    call get_det(det)

END SUBROUTINE getJacobianDet

SUBROUTINE get_elm_data(iqpt, nel1, DIM1, ecrd, evel, erot)

USE lat_ori_fem_mod
USE IntrinsicTypesModule, RK=>REAL_KIND

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: iqpt
    INTEGER, INTENT(IN) :: nel1
    INTEGER, INTENT(IN) :: DIM1

    REAL(RK), INTENT(OUT) :: ecrd(0:nel1, 0:DIM1)
    REAL(RK), INTENT(OUT) :: evel(0:nel1, 0:DIM1)
    REAL(RK), INTENT(OUT) :: erot(0:nel1, 0:DIM1)

    call getInterestingNodes(erot, ecrd, evel, iqpt)


END SUBROUTINE get_elm_data

SUBROUTINE get_disc_dens(nel1, nc1, DIM1, rot, density)

USE lat_ori_fem_mod
USE IntrinsicTypesModule, RK=>REAL_KIND

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: nel1
    INTEGER, INTENT(IN) :: nc1
    INTEGER, INTENT(IN) :: DIM1

    REAL(RK), INTENT(IN) :: rot(0:nc1, 0:DIM1)

    REAL(RK), INTENT(OUT) :: density(0:11, 0:nel1)

    call get_dislocation_density(rot, density)

END SUBROUTINE get_disc_dens

SUBROUTINE deallocate_vars()

    USE lat_ori_fem_mod

    IMPLICIT NONE

    CALL deallocate_data()

END SUBROUTINE deallocate_vars

end module mainLatProgram
