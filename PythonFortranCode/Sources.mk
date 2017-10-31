LIBBASE = orifem
LIBRARY = lib$(LIBBASE).a

f90SOURCES=\
IntrinsicTypesModule.f90\
units.f90\
ConstantsModule.f90\
quadrature.f90\
matrixMath.f90\
shape_3d.f90\
femVariables.f90\
schmidTensor.f90\

mSOURCES=\
LatOriFEM.f90\

f90OBJECTS=$(f90SOURCES:.f90=.o)
mOBJECTS=$(mSOURCES:.f90=.o)