# Tutorial: Computer Vision and Machine Learning with Python, Keras and OpenCV

### Includes a demonstration of concepts with Gesture Recognition.
This was created as part of an educational for the [Western Founders Network](https://foundersnetwork.ca/) computer vision and machine learning educational session.

## Demo

The final demo can be seen [here](https://www.youtube.com/watch?v=IJV11OGTNT8) and below:

<a href="https://imgflip.com/gif/22n3o6"><img src="https://i.imgflip.com/22n3o6.gif"/></a>

## Contents
[notebook.ipynb](https://github.com/jrobchin/Computer-Vision-Basics-with-Python-Keras-and-OpenCV/blob/master/notebook.ipynb) contains a full tutorial of basic computer vision and machine learning concepts, including:

* *What computers see*
* Image Filters and Functions
  - Blurring
  - Dilating
  - Erosion
  - Canny Edge Detectors
  - Thresholding
* Background Subtraction Techniques
  - Using a background image to find differences
  - Using motion based background subtraction algorithms
* Contours
  - Finding and sorting contours
* Tracking
* (Deep) Neural Networks 
* (Deep) Convolutional Neural Networks
* Demo Project: Gesture Recognition
  - Extracting the subject
  - Tracking the hand
  - Collecting data
  - Building the Neural Network
  - Preparing Data for Training
  - Training the Network
  - Plotting Model History
  
*Note: Please check the [issues](https://github.com/jrobchin/Computer-Vision-Basics-with-Python-Keras-and-OpenCV/issues) on this repo if you're having problems with the notebook.*

## Installation Instructions ('$' means run this in terminal/command prompt, do not type '$')
### Windows:
* Install Anaconda (https://www.anaconda.com/download/) or Miniconda (https://conda.io/miniconda.html) to save hard drive space
* Start an Anaconda Prompt. (Search Anaconda in the start menu.)
### Mac/Linux: Manually installing packages
* Install Anaconda (https://www.anaconda.com/download/) or Miniconda (https://conda.io/miniconda.html) to save hard drive space
#### Mac:
* For Miniconda, open terminal and navigate to the directory you downloaded Miniconda3-latest-MacOSX-x86_64.sh to and run:

        $ bash Miniconda3-latest-MacOSX-x86_64.sh

* For Anaconda, double click the Anaconda3-5.0.1-MacOSX-x86_64.pkg file you downloaded

#### Linux:
* For Miniconda, open a terminal and navigate to the directory you downloaded Miniconda3-latest-Linux-x86_64.sh to and run:

        $ bash Miniconda3-latest-Linux-x86_64.sh

* For Anaconda, open a terminal and navigate to the directory you downloaded Anaconda3-5.0.1-Linux-x86_64.sh to and run:

        $ bash Anaconda3-5.0.1-Linux-x86_64.sh

### All Platforms:
* Create and activate a Python 3.5 conda environment called cv.

        $ conda create -n cv python=3.5

        $ source activate cv
* Install Numpy (http://www.numpy.org/)

        $ conda install numpy
* Install Matplotlib (https://matplotlib.org/)

        $ conda install matplotlib
* Install Keras (https://keras.io/) 

        $ conda install keras
    - This should also install tensorflow
* Install h5py (http://www.h5py.org/)

        $ conda install h5py
* Install Jupyter Notebook (http://jupyter.org/)

        $ conda install jupyter notebook
* Install IPython (https://ipython.org/)

        $ conda install ipython
* Install OpenCV3 (https://opencv.org/)
        
        $ conda install -c conda-forge opencv 
    
    if the `import cv2` does not work with this install, try instead:
    
        $ conda install -c https://conda.anaconda.org/menpo opencv3
