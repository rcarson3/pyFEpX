# pyFEpX
Python tools to process FEpX data and some OdfPF capabilities built in

You will need scipy, numpy, and the academic version of iopro obtained from Anaconda.
The easiest way to install these is using the Anaconda software conda. In order to use iopro, you will need to sign up on their academic website. Then you will want to use Python 3.5 for all of these packages until iopro is updated to 3.6. If you don't want to use iopro you can substitute the iopro.genfromtxt calls with the equivalent call np.genfromtxt or numpy.genfromtxt inside the FePX_Data_and_Mesh.py file. I would recommend getting iopro due to it being vastly faster over the numpy equivalent call.

You can also use https://github.com/ContinuumIO/TextAdapter as a replacement to iopro. It's the renamed and open sourced version of iopro.

I've also include a few example scripts to process the FEpX data.

Included in the OdfPF package are complete versions of the rotation and misorientation modules from the matlab code. The creation and examination of pole figures and inverse pole figures are still a work in progress. 

Contains a modified version of Paulo Herrera's PyEVTK library found at https://bitbucket.org/pauloh/pyevtk/overview.
The PyEVTK library allows for the creation of binary VTK files to be used in either VisIT or Paraview.


