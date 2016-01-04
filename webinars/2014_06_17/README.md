# Introduction to Python GPU Programming with Numba and NumbaPro

*NOTE: This notebook has been modified from the version used in the Webinar, in
order to keep it current with the supported Numba and Accelerate APIs following
the deprecation of NumbaPro. It was last updated for Numba 0.22.1 and Accelerate
2.0*

This directory contains the ipython notebook used in the Continuum webinar on
June 17, 2014.  The notebook is created on gpu.wakari.io, a preview version of 
the next version of Wakari that is temporarily made available from June 17, 2014
to June 20, 2014.  The outputs in the notebook is obtained by executing on the 
Wakari platform in an Amazon GPU instance.

## Viewing in Slide Mode

To view the ipython notebook in slidemode.

1. Go to http://slideviewer.herokuapp.com 
2. Copy this link: https://raw.githubusercontent.com/ContinuumIO/numbapro-examples/master/webinars/2014_06_17/intro_to_gpu_python.ipynb
3. Paste it on the input box
4. Submit the form

## Running the Notebook on Your Machine

If you have a computer with a CUDA GPU with compute capability >= 2.0,
you can run the webinar on your machine.

### Install requirements

1. Download Anaconda: http://continuum.io/downloads
2. Once Anaconda is installed, you will have access to ``conda`` in the terminal
3. Install Accelerate in terminal ``$ conda install accelerate``.  Details about Accelerate: https://store.continuum.io/cshop/accelerate/
4. You should be able ``import accelerate`` in python now
5. Get Jupyter notebook: ``$ conda install jupyter``
6. Launch ``$ jupyter notebook`` in the directory containing the notebook.





