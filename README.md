# Note: This repo was related to the old Cornell DPLab FEpX code. I am no longer maintaining this. I am now focusing on the development of the open-source LLNL ExaConstit and ExaCMech crystal plasticity modeling and finite element libraries/application codes. Since these libraries are being heavily focused on running on next-gen exascale computers such as Frontier and El Capitan.    

# pyFEpX
Python tools to process FEpX data and some OdfPF capabilities built in. The library provides a number of features including orientation convertions, misorientation metrics, orientation-spatial metrics, intragrain deformation metrics, superconvergence methods, dislocation density methods, and finally binary vtk preparation scripts. 

You will need scipy, numpy, and TextAdapter obtained from Anaconda.
The easiest way to install these is using the Anaconda software conda. You can obtain TextAdapter at https://github.com/ContinuumIO/TextAdapter as a replacement to iopro. It's the renamed and open sourced version of iopro. It does include some installations steps on your own.

I've also include a few example scripts to process the FEpX data. I'm open to also including support for other simulation result readers and conversions processess from their method to the ones used here.

Included in the OdfPF package are complete versions of the rotation and misorientation modules from the matlab code. The creation and examination of pole figures and inverse pole figures are still a work in progress.

The intragrain deformation metrics provided in https://doi.org/10.1088/1361-651X/aa6dc5 are also provided under the FiniteElement.lofem_elas_stretch_stats and FiniteElement.deformationStats functions.



Contains a modified version of Paulo Herrera's PyEVTK library found at https://bitbucket.org/pauloh/pyevtk/overview. The license for PyEVTK hosted here can be found at https://github.com/rcarson3/pyFEpX/blob/master/PythonScripts/pyevtk/src/LICENSE  
The PyEVTK library allows for the creation of binary VTK files to be used in either VisIT or Paraview.


