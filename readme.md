## Intro to neural network architectures

This tutorial was given at the meeting of the Munich PyData group on 3.5.2018.  
It contains three sections:
* Dense layers
* Convolutional layers
* Recurrent layers

Although a notebook, this is a talk and best viewed with the RISE plugin. All comments were marked as skip, and to satisfy the need of a talk a lot of images were imported.  The orginal notebook is available, but the much improved version is Intro_to_neural_networks.ipynb, where comments has been added and the weights of the models are supplied as well, as some require some training time on a laptop.  

It was not intended to create high performance models, in particular the models tend to generalize rather badly as no dropout has been used.  


The notebook was developed with a standard andromeda python 3.6 distribution, for the tutorial the MNIST dataset and Keras was used. 
Also needed is a graphviz installation to visualize the models, check the requirements of the keras.utils.vis_utils package.  

The model visualisation in the 'vis' folder is in improved version of the plot_model routine from keras.  Is work in progress and will be published as a separate GitHub project.  
