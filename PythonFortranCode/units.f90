MODULE units_mod

  USE IntrinsicTypesModule

  PUBLIC

  !INTEGER, PARAMETER :: NDIM  = 8 ! 8-noded brick
  !INTEGER, PARAMETER :: NDIM  = 4 ! 4-noded tetrahedra
  INTEGER, PARAMETER :: NDIM  = 10 ! 10-noded tetrahedra
  INTEGER, PARAMETER :: NNPE  = NDIM - 1

  INTEGER, PARAMETER :: KDIM  = 3*NDIM
  INTEGER, PARAMETER :: KDIM1 = KDIM - 1

  INTEGER, PARAMETER :: DIMS1 = 2

  INTEGER :: nelem1, ncrd1, ngdot1, ncvec1

END MODULE units_mod
