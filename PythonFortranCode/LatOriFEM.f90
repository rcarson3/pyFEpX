module lat_ori_fem_mod

USE IntrinsicTypesModule, RK=>REAL_KIND
USE quadrature_mod
USE shape_3d_mod
USE units_mod
USE femVariables
USE matrixMath

IMPLICIT NONE

PRIVATE :: estrain, gdot, devdp, W, devwp
    REAL(RK), ALLOCATABLE :: estrain(:,:,:), gdot(:,:), devdp(:,:,:), &
                                 & W(:,:,:), devwp(:,:,:)

CONTAINS

SUBROUTINE initializedata(nelem, ncrd, ngdot)

    USE schmidTensor

    IMPLICIT NONE

    INTEGER, INTENT(IN) :: nelem, ncrd, ngdot

    nelem1 = nelem - 1
    ncrd1 = ncrd - 1
    ngdot1 = ngdot - 1
    ncvec1 = ncrd*3 - 1

    ALLOCATE(estrain(0:nelem1, 0:DIMS1, 0:DIMS1), gdot(0:nelem1, 0:ngdot1), &
             &devdp(0:nelem1, 0:DIMS1, 0:DIMS1),W(0:nelem1, 0:DIMS1, 0:DIMS1), &
             &devwp(0:nelem1, 0:DIMS1, 0:DIMS1))


    estrain = 0.0_RK
    gdot = 0.0_RK
    devdp = 0.0_RK
    W = 0.0_RK
    devwp = 0.0_RK

    call init_femvar()

    call initquad()

    call setSchmidTensor()

END SUBROUTINE initializedata

SUBROUTINE initquad()

    IMPLICIT NONE

    call initialize()

END SUBROUTINE initquad

SUBROUTINE getInterestingNodes(erot, ecrd, evel, iqpt)

    IMPLICIT NONE
    REAL(RK), INTENT(OUT) :: erot(0:nelem1, 0:DIMS1), ecrd(0:nelem1, 0:DIMS1),&
                             &evel(0:nelem1, 0:DIMS1)
    INTEGER, INTENT(IN) :: iqpt

    call set_i2()
    call get_vec_node_vals(grot, erot, iqpt)
    call get_elm_node_vals(vel, evel, iqpt)
    call get_elm_node_vals(coord, ecrd, iqpt)

END SUBROUTINE getInterestingNodes

SUBROUTINE get_vec_node_vals(node, elmval, iqpt)

    IMPLICIT NONE
    REAL(RK), INTENT(IN) :: node(0:ncvec1)
    REAL(RK), INTENT(OUT) :: elmval(0:nelem1, 0:DIMS1)
    INTEGER, INTENT(IN) :: iqpt

    REAL(RK) :: elmnode(0:nelem1, 0:KDIM1), qpnt(0:DIMS1)
    !REAL(RK) :: tmpelm(0:9, 0:DIMS1)
    REAL(RK) :: N(0:DIMS1, 0:KDIM1), NT(0:KDIM1, 0:DIMS1)
    INTEGER :: i, j

    elmval = 0.0_RK

    call getElemVec(node, elmnode)

    qpnt(0) = qploc(0, iqpt)
    qpnt(1) = qploc(1, iqpt)
    qpnt(2) = qploc(2, iqpt)

    do i = 0,nelem1

        call get_sf_array(N, NT, iqpt)
        elmval(i,:) = MATMUL(elmnode(i,:),NT)

    end do


END SUBROUTINE get_vec_node_vals

SUBROUTINE get_elm_node_vals(node, elmval, iqpt)

    IMPLICIT NONE
    REAL(RK), INTENT(IN) :: node(0:ncrd1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: elmval(0:nelem1, 0:DIMS1)
    INTEGER, INTENT(IN) :: iqpt

    REAL(RK) :: elmnode(0:nelem1, 0:KDIM1), qpnt(0:DIMS1)
    !REAL(RK) :: tmpelm(0:9, 0:DIMS1)
    REAL(RK) :: N(0:DIMS1, 0:KDIM1), NT(0:KDIM1, 0:DIMS1)
    INTEGER :: i, j

    elmval = 0.0_RK

    call getElemCoord(node, elmnode)

    do i = 0,nelem1

        call get_sf_array(N, NT, iqpt)
        elmval(i,:) = MATMUL(elmnode(i,:),NT)

    end do


END SUBROUTINE get_elm_node_vals

SUBROUTINE get_def_grad(def_grad, disp, det, elmval, iqpt, cenval)

    IMPLICIT NONE

    REAL(RK), INTENT(OUT) :: def_grad(0:nelem1, 0:DIMS1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: det(0:nelem1) !the determinate for each element as change from local element and parent element
    LOGICAL, INTENT(IN) :: elmval  ! if you only care about elemental value
    INTEGER, INTENT(IN) :: iqpt ! if above true this pretty much ignored
    REAL(RK), INTENT(IN) :: disp(0:ncrd1, 0:DIMS1)
    LOGICAL, INTENT(IN) :: cenval ! if you want only the centroid value not available if doing elem val

    REAL(RK) :: dndx(0:nelem1, 0:nnpe)
    REAL(RK) :: dndy(0:nelem1, 0:nnpe)
    REAL(RK) :: dndz(0:nelem1, 0:nnpe)
    REAL(RK) :: loc0, loc1, loc2
    REAL(RK) :: t_grad(0:nelem1, 0:DIMS1, 0:DIMS1)
    !inverted jacobian matrix values
    REAL(RK) :: s11(0:nelem1), s12(0:nelem1), s13(0:nelem1)
    REAL(RK) :: s21(0:nelem1), s22(0:nelem1), s23(0:nelem1)
    REAL(RK) :: s31(0:nelem1), s32(0:nelem1), s33(0:nelem1)
    REAL(RK) :: elmCoord(0:nelem1, 0:KDIM1)
    REAL(RK) :: elmDisp(0:nelem1, 0:KDIM1)
    INTEGER :: iqpt1, i

!    REAL(RK) :: wt

    def_grad = 0.0_RK

    call getElemCoord(coord, elmCoord)
    call getElemCoord(disp, elmDisp)


    if (elmval) then

        do iqpt1 = 0, nqpt1

            ! coordinates in the parent element
            loc0 = qploc(0, iqpt1)
            loc1 = qploc(1, iqpt1)
            loc2 = qploc(2, iqpt1)
            ! weigth
!            wt = wtqp(0, iqpt)

            call sfder_hpar(loc0, loc1, loc2, elmCoord, dndx, dndy,  &
            &     dndz, det, s11, s12, s13, s21, s22, s23, s31, s32,  &
            &     s33)

            call mat_gradient(t_grad, dndx, dndy, dndz, elmDisp)


            def_grad = def_grad + (1.0_RK/(nqpt1+1.0_RK))*t_grad

        end do

        def_grad(:, 0, 0) = def_grad(:, 0, 0) + 1
        def_grad(:, 1, 1) = def_grad(:, 1, 1) + 1
        def_grad(:, 2, 2) = def_grad(:, 2, 2) + 1


    else
       if (cenval) then

          loc0 = 0.25_RK 
          loc1 = 0.25_RK
          loc2 = 0.25_RK

       else

          loc0 = qploc(0, iqpt)
          loc1 = qploc(1, iqpt)
          loc2 = qploc(2, iqpt)
       endif
       ! weigth

       call sfder_hpar(loc0, loc1, loc2, elmCoord, dndx, dndy,  &
            &     dndz, det, s11, s12, s13, s21, s22, s23, s31, s32,  &
            &     s33)

       call mat_gradient(def_grad, dndx, dndy, dndz, elmDisp)

       def_grad(:, 0, 0) = def_grad(:, 0, 0) + 1
       def_grad(:, 1, 1) = def_grad(:, 1, 1) + 1
       def_grad(:, 2, 2) = def_grad(:, 2, 2) + 1
        
    endif

END SUBROUTINE get_def_grad

SUBROUTINE get_dislocation_density(rot, density)
    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: rot(0:ncrd1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: density(0:11, 0:nelem1)
    REAL(RK) :: lmat(12, 9)
    REAL(RK) :: alpha(0:8, 0:nelem1)

    REAL(RK), PARAMETER :: a = sqrt(3.0)/9.0
    REAL(RK), PARAMETER :: b = sqrt(3.0)/84.0
    REAL(RK), PARAMETER :: c = 1.0/18.0
    REAL(RK), PARAMETER :: d = 3.0/14.0
    REAL(RK), PARAMETER :: z = 0.0

    INTEGER :: i



    lmat(:, 1) = (/a, -a, z, a, -a, z, a, -a, z, a, -a, z/)
    lmat(:, 2) = (/7.0*c, 13.0*c, c, -7.0*c, -13.0*c, -c, -7.0*c, -13.0*c, -c, 7.0*c, 13.0*c, c/)
    lmat(:, 3) = (/-13.0*c, -7.0*c, -c, 13.0*c, 7.0*c, c,-13.0*c, -7.0*c, -c, 13.0*c, 7.0*c, c/)
    lmat(:, 4) = (/7.0*c, c, 13.0*c, -7.0*c, -c, -13.0*c, -7.0*c, -c, -13.0*c, 7.0*c, c, 13.0*c/)
    lmat(:, 5) = (/-a, z, a, -a, z, a, -a, z, a, -a, z, a/)
    lmat(:, 6) = (/13.0*c, c, 7.0*c, 13.0*c, c, 7.0*c, -13.0*c, -c, -7.0*c, -13.0*c, -c, -7.0*c/)
    lmat(:, 7) = (/c, 7.0*c, 13.0*c, -c, -7.0*c, -13.0*c, c, 7.0*c, 13.0*c, -c, -7.0*c, -13.0*c/)
    lmat(:, 8) = (/-c, -13.0*c, -7.0*c, -c, -13.0*c, -7.0*c, -c, 13.0*c, 7.0*c, c, 13.0*c, 7.0*c/)
    lmat(:, 9) = (/z, a, -a, z, a, -a, z, a, -a, z, a, -a/)


    call get_alpha_tensor(alpha, rot)

    do i = 0, nelem1
        density(:, i) = matmul(lmat, alpha(:, i))
    enddo


END SUBROUTINE get_dislocation_density

SUBROUTINE get_alpha_tensor(grad, vec)

    IMPLICIT NONE

    REAL(RK), INTENT(OUT) :: grad(0:DIMS1, 0:DIMS1, 0:nelem1)
    REAL(RK), INTENT(IN) :: vec(0:ncrd1, 0:DIMS1)

    REAL(RK) :: dndx(0:nelem1, 0:nnpe)
    REAL(RK) :: dndy(0:nelem1, 0:nnpe)
    REAL(RK) :: dndz(0:nelem1, 0:nnpe)
    REAL(RK) :: loc0, loc1, loc2
    REAL(RK) :: t_grad(0:nelem1, 0:DIMS1, 0:DIMS1)
    !inverted jacobian matrix values
    REAL(RK) :: s11(0:nelem1), s12(0:nelem1), s13(0:nelem1)
    REAL(RK) :: s21(0:nelem1), s22(0:nelem1), s23(0:nelem1)
    REAL(RK) :: s31(0:nelem1), s32(0:nelem1), s33(0:nelem1), det(0:nelem1)
    REAL(RK) :: elmCoord(0:nelem1, 0:KDIM1)
    REAL(RK) :: elmVec(0:nelem1, 0:KDIM1), trk(0:nelem1)
    INTEGER :: iqpt1, i

!    REAL(RK) :: wt

    grad = 0.0_RK

    call getElemCoord(coord, elmCoord)
    call getElemCoord(vec, elmVec)

    loc0 = 0.25_RK
    loc1 = 0.25_RK
    loc2 = 0.25_RK

    call sfder_hpar(loc0, loc1, loc2, elmCoord, dndx, dndy,  &
        &     dndz, det, s11, s12, s13, s21, s22, s23, s31, s32,  &
        &     s33)

    call mat_gradientT(grad, dndx, dndy, dndz, elmVec)

    trk(:) = grad(0,0,:) + grad(1,1,:) + grad(2,2,:)
    grad(0,0,:) = grad(0,0,:) - trk(:)
    grad(1,1,:) = grad(1,1,:) - trk(:)
    grad(2,2,:) = grad(2,2,:) - trk(:)

END SUBROUTINE get_alpha_tensor

SUBROUTINE get_vel_grad(vel_grad, det, elmval, iqpt)

    IMPLICIT NONE

    REAL(RK), INTENT(OUT) :: vel_grad(0:nelem1, 0:DIMS1, 0:DIMS1)
    REAL(RK), INTENT(OUT) :: det(0:nelem1) !the determinate for each element as change from local element and parent element
    LOGICAL, INTENT(IN) :: elmval  ! if you only care about elemental value
    INTEGER, INTENT(IN) :: iqpt ! if above true this pretty much ignored

    REAL(RK) :: dndx(0:nelem1, 0:nnpe)
    REAL(RK) :: dndy(0:nelem1, 0:nnpe)
    REAL(RK) :: dndz(0:nelem1, 0:nnpe)
    REAL(RK) :: loc0, loc1, loc2
    REAL(RK) :: t_grad(0:nelem1, 0:DIMS1, 0:DIMS1)
    !inverted jacobian matrix values
    REAL(RK) :: s11(0:nelem1), s12(0:nelem1), s13(0:nelem1)
    REAL(RK) :: s21(0:nelem1), s22(0:nelem1), s23(0:nelem1)
    REAL(RK) :: s31(0:nelem1), s32(0:nelem1), s33(0:nelem1)
    REAL(RK) :: elmCoord(0:nelem1, 0:KDIM1), elmVel(0:nelem1, 0:KDIM1)

    INTEGER :: iqpt1

!    REAL(RK) :: wt

    call getElemCoord(coord, elmCoord)
    call getElemCoord(vel, elmVel)

    vel_grad = 0.0_RK

    det = 0.0_RK

    if (elmval) then

        do iqpt1 = 0, nqpt1

            ! coordinates in the parent element
            loc0 = qploc(0, iqpt1)
            loc1 = qploc(1, iqpt1)
            loc2 = qploc(2, iqpt1)
            ! weigth
!            wt = wtqp(0, iqpt)

            call sfder_hpar(loc0, loc1, loc2, elmCoord, dndx, dndy,  &
            &     dndz, det, s11, s12, s13, s21, s22, s23, s31, s32,  &
            &     s33)

            call mat_gradient(t_grad, dndx, dndy, dndz, elmVel)

            vel_grad = vel_grad + (1.0_RK/(nqpt1+1.0_RK))*t_grad

        enddo

    else

        loc0 = qploc(0, iqpt)
        loc1 = qploc(1, iqpt)
        loc2 = qploc(2, iqpt)
        ! weigth

        call sfder_hpar(loc0, loc1, loc2, elmCoord, dndx, dndy,  &
        &     dndz, det, s11, s12, s13, s21, s22, s23, s31, s32,  &
        &     s33)

        call mat_gradient(vel_grad, dndx, dndy, dndz, elmVel)


    endif

END SUBROUTINE get_vel_grad

SUBROUTINE mat_gradient(ndgrad, dndx, dndy, dndz, gnd)
    IMPLICIT NONE
    !
    !     Compute 3d vector gradient.
    !
    !----------------------------------------------------------------------
    !
    !     Arguments:
    !
    REAL(RK), INTENT(OUT) :: ndgrad(0:nelem1, 0:DIMS1, 0:DIMS1)
    REAL(RK), INTENT(IN)  :: dndx(0:nelem1, 0:nnpe)
    REAL(RK), INTENT(IN)  :: dndy(0:nelem1, 0:nnpe)
    REAL(RK), INTENT(IN)  :: dndz(0:nelem1, 0:nnpe)
    REAL(RK), INTENT(IN)  :: gnd(0:nelem1, 0:kdim1)
    !
    !     Locals:
    !    
    INTEGER :: i, i1, i2, i3
    !
    !----------------------------------------------------------------------
    !
    ndgrad = 0.0_RK

    !This one should be now faster than the version in fepx because of correct striding

    do i = 0, nnpe

        i1 = 3 * i
        i2 = i1 + 1
        i3 = i2 + 1

        ndgrad(:, 0, 0) = ndgrad(:, 0, 0) + dndx(:, i) * gnd(:, i1)
        ndgrad(:, 0, 1) = ndgrad(:, 0, 1) + dndy(:, i) * gnd(:, i1)
        ndgrad(:, 0, 2) = ndgrad(:, 0, 2) + dndz(:, i) * gnd(:, i1)

        ndgrad(:, 1, 0) = ndgrad(:, 1, 0) + dndx(:, i) * gnd(:, i2)
        ndgrad(:, 1, 1) = ndgrad(:, 1, 1) + dndy(:, i) * gnd(:, i2)
        ndgrad(:, 1, 2) = ndgrad(:, 1, 2) + dndz(:, i) * gnd(:, i2)

        ndgrad(:, 2, 0) = ndgrad(:, 2, 0) + dndx(:, i) * gnd(:, i3)
        ndgrad(:, 2, 1) = ndgrad(:, 2, 1) + dndy(:, i) * gnd(:, i3)
        ndgrad(:, 2, 2) = ndgrad(:, 2, 2) + dndz(:, i) * gnd(:, i3)

    enddo

END SUBROUTINE mat_gradient

SUBROUTINE mat_gradientT(ndgrad, dndx, dndy, dndz, gnd)
IMPLICIT NONE
!
!     Compute 3d vector gradient transpose.
!
!----------------------------------------------------------------------
!
!     Arguments:
!
REAL(RK), INTENT(OUT) :: ndgrad(0:DIMS1, 0:DIMS1, 0:nelem1)
REAL(RK), INTENT(IN)  :: dndx(0:nelem1, 0:nnpe)
REAL(RK), INTENT(IN)  :: dndy(0:nelem1, 0:nnpe)
REAL(RK), INTENT(IN)  :: dndz(0:nelem1, 0:nnpe)
REAL(RK), INTENT(IN)  :: gnd(0:nelem1, 0:kdim1)
!
!     Locals:
!
INTEGER :: i, j, i1, i2, i3
!
!----------------------------------------------------------------------
!
ndgrad = 0.0_RK

!Gives the transpose of the gradient
do j = 0, nnpe
    do i = 0, nelem1

        i1 = 3 * j
        i2 = i1 + 1
        i3 = i2 + 1

        ndgrad(0, 0, i) = ndgrad(0, 0, i) + dndx(i, j) * gnd(i, i1)
        ndgrad(0, 1, i) = ndgrad(0, 1, i) + dndy(i, j) * gnd(i, i2)
        ndgrad(0, 2, i) = ndgrad(0, 2, i) + dndz(i, j) * gnd(i, i3)

        ndgrad(1, 0, i) = ndgrad(1, 0, i) + dndx(i, j) * gnd(i, i1)
        ndgrad(1, 1, i) = ndgrad(1, 1, i) + dndy(i, j) * gnd(i, i2)
        ndgrad(1, 2, i) = ndgrad(1, 2, i) + dndz(i, j) * gnd(i, i3)

        ndgrad(2, 0, i) = ndgrad(2, 0, i) + dndx(i, j) * gnd(i, i1)
        ndgrad(2, 1, i) = ndgrad(2, 1, i) + dndy(i, j) * gnd(i, i2)
        ndgrad(2, 2, i) = ndgrad(2, 2, i) + dndz(i, j) * gnd(i, i3)

    enddo
enddo

END SUBROUTINE mat_gradientT

SUBROUTINE get_sf_array(N, NT, iqpt)

    IMPLICIT NONE

    REAL(RK), INTENT(OUT) :: N(0:DIMS1, 0:KDIM1), NT(0:KDIM1, 0:DIMS1)
    INTEGER, INTENT(IN) :: iqpt ! Only forms N at the current iqpt
    INTEGER :: innpe, i1
    REAL(RK) :: loc0, loc1, loc2, sf(0:nnpe)

    loc0 = qploc(0, iqpt)
    loc1 = qploc(1, iqpt)
    loc2 = qploc(2, iqpt)
    call T10_shape_hpar(loc0, loc1, loc2, sf)

    N = 0.0_RK
    NT = 0.0_RK

    do innpe = 0,nnpe

        i1 = innpe*3
        !Not really a good way to do this using strides
        N(0, i1) = sf(innpe)
        N(1, (i1+1)) = sf(innpe)
        N(2, (i1+2)) = sf(innpe)

        NT(i1, 0) = sf(innpe)
        NT((i1+1), 1) = sf(innpe)
        NT((i1+2), 2) = sf(innpe)

    end do



END SUBROUTINE get_sf_array

SUBROUTINE setDataUp(eps, gammadot, velocity, connectivity, crd, rot)

    IMPLICIT NONE

    REAL(RK), INTENT(IN) :: eps(0:nelem1, 0:DIMS1, 0:DIMS1), gammadot(0:nelem1, 0:ngdot1)
    REAL(RK), INTENT(IN) :: velocity(0:ncrd1, 0:DIMS1), rot(0:ncvec1), crd(0:ncrd1, 0:DIMS1)
    INTEGER, INTENT(IN) :: connectivity(0:nelem1, 0:NNPE)

    estrain = eps
    gdot = gammadot
    vel = velocity
    coord = crd
    connect = connectivity
    grotnot = rot
    grot = rot

END SUBROUTINE setDataUp

SUBROUTINE returnData(onedarray)

    IMPLICIT NONE

    REAL(RK), INTENT(OUT) :: onedarray(0:ncvec1)

    onedarray = 1.0_RK

    onedarray = grot

!    call cleanup_petsc()

END SUBROUTINE returnData

SUBROUTINE deallocate_data()

    USE schmidTensor

    IMPLICIT NONE

    deallocate(estrain, gdot, devdp, W, devwp)
    call deallocate_femvar()
    call deallocate_schmid()

END SUBROUTINE deallocate_data

end module lat_ori_fem_mod
