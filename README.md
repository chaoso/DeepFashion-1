# DeepFashion


## Description
This is a python implementation of the paper [Be Your Own Prada: Fashion Synthesis with Structural Coherence](https://arxiv.org/pdf/1710.07346.pdf), using the Pytorch library.

DeepFashion is a project that focuses on creating fashion synthesis with structural coherence. This is done by using a stackedGAN together with a text encoder, which is build on a RNN architecture. 

## Acknowledgement
The inspiration for this work is based on the paper Be Your Own Prada: Fashion Synthesis with Structural Coherence by Zhuet.  al. Addtionally, the data used in this work is from the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html).  

## Dependencies
Since the data used for this project has size of 17 GB, we have not uploaded it on Github. Please download the data on the following link: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/FashionSynthesis.html and put them into a "Data" Folder in the root directory.

## How to run
The Network can be run by running the Main.ipynb.
On the first run it is important to run the pre-processing code first. Be aware this may take up to a hour. If you wish to achieve results similair to the ours then the both GAN's need to be re-trained, which may take 24 hours per GAN.
