# RHGAN

## Abstract 

Generative Adversarial Networks are one of the most recent powerful machine learning models that generate initiating realistic data from a given data set. Generative adversarial networks were initially designed and introduced by Ian Goodfellow in June 2014. They are simply neural networks class of machine learning frameworks. They were presented with the intention of overcoming the data availability issues that arise with deep neural networks DNNs. DNNs have been used widely among various domains and applications. These networks rely heavily on massive sets of labeled training data which is either unavailable or costly to obtain. 

GANs provide the possibility of developing and generating richer data sets with training on very small labeled data sets.  GANs achieve a high level of realism in generating realistic data sets by combining two neural network models. A generative model that attempts to create realistic data sets by being trained on given labeled data sets, and a discriminator model that determines if the output data is synthetic or belongs to the given training data.


## About
 
- Project Type: Capstone Project 
- University: UMass Boston
- Department: College of Math and Science 
- Course: CS410 Introduction to Software Engineering 
- Term: Spring 2022 
- Supervised by: Professor Daniel Haehn
- Developed for: Client Ricarda Haehn 
- Developed by: 
    - Zara El Alaoui 
    - Vladimir Pierre Louis 
    - Sara Saddighi 
    - Zhuoping Chen
    - Jessica Carvalho

## Purpose

Our final project is a fully back-end based application that takes in a dataset as an input and returns
a generated artwork as an output. The application supports a variety of themes which are listed as
the following:

- Demons
- Hell
- Psychedelic
- Purgatory
- Dark
- Depression
- All of the themes together

The application takes user input for the theme selection as well as for the desired output size, as our
application supports two sizes which are:

- One picture
- A 7 x 7 collection of pictures

The application then uses the corresponding NPY file as training dataset as an input and generates
artwork as an output. 

## Technologies

- Python: The GAN neural network was built using Python which is a high-level, interpreted, general-purpose programing language. 

- PyPI: For installing most of the python libraries needed, we used PyPI which is a package manager for python packages and modules. 

- Tensorflow: We utilized the Tensorflow library which is an open-source framework machine learning and artificial intelligence. Tensorflow is the main library that we used to construct and build both of the neural networks that construct our GAN: the generator and the discriminator. 

- React & JavaScript: Since we also developed a project portfolio website, we used React and Javascript for a modern user interface. Our project portfolio website is hosted on Github Pages which is a free web hosting service provided by GitHub. We used gh-pages which is an NPM package that allowed us to do the process. Our website provides information about our project, and links for the documentation, presentation slides, and the Github repository for the codebase, as well as links to LinkedIn profiles for all team members.

- Conda: We used conda to set up our environement. Conda is an open source package management system and environement management system. 


## Links 
- Project Portfolio Website: https://zara-18-z.github.io/RHGAN-PortfolioWebsite/
- Presentation Slides: https://liveumb-my.sharepoint.com/:p:/g/personal/f_elalaoui001_umb_edu/EQzkWwT5KdNAjkHEPUSSQ-oBWR9fQmBLNwr5wz0wqMgN9Q?e=VSzy1X
 
## Hardware 

To be able to run this project, this software relies on the user having good hardware (both GPU and CPU) to perform the training and generate the images. We were given access through our university to use Chimera which is a heterogeneous distributed memory high performance compute cluster, comprised of a head node and 12 compute nodes. 

## Environment Setup 

Before you are able to run this project you need to first setup the proper environment on your computer. 

- Installing Python: The first thing you need to do is to install python on your Machine. (If you already have python installed you can skip this step):
    - Go to Python’s download page and download the appropriate installer for your machine.
    - Make sure to add python to your path when prompted.
   
- Installing Conda: 
    - https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
- Installing Tensorflow: 
    - https://www.tensorflow.org/install
    
Once you have conda, you can easily set up the required environement to run this project by using our [Environement.yaml] file which can be used to create the same environement as the one we used to run develop and run this project. You can run the following command to use the environement:
```Python
Conda env create -f environement.yaml 
```

## Code Documentation 
To run the artwork generator software you should follow the following steps: 
- First thing you should do is donwload one of the following datasets from this DropBox link or the Mega link and put the folder titeled [Data_NPY_Files] in the same directory as the GAN.py file: 

    - [DropBox](https://www.dropbox.com/s/j136jhc8121xl97/Data_NPY_Files.zip?dl=0)
    - [Mega](https://mega.nz/folder/QD8gWAQD#lIsT2P6f7mmEKnthlkGDxg)
- You should then navigate to the RH_GAN folder where the GAN.py is located. 

     ```Python 
     cd RH_GAN 
     ```
 - You should then activate the conda environement where you installed all the python packages required: 
 
      ```Python 
     conda activate environement_name
     ```

- The next step is to run the GAN as the following: 

     ```Python 
     python GAN.py 
     ```
- Once you run the GAN, you will get asked for a theme input as the following: 

    ```Python 
      Hello! Welcome To Your Favorite Art Generator!
      Get Ready To Build Something Amazing With RHGAN!
      Here is a list of the themes we have:
       1: Demons
       2: Hell
       3: Psychedelic
       4: Purgatory
       5: Dark
       6: Depression
       7: All of these themes together
      Please make a selection!
    ```
- As soon as you make a theme selection, you will get promoted to make a size selection as the following: 

     ```Python 
      Hello! Welcome To Your Favorite Art Generator!
      Get Ready To Build Something Amazing With RHGAN!
      Here is a list of the themes we have:
       1: Demons
       2: Hell
       3: Psychedelic
       4: Purgatory
       5: Dark
       6: Depression
       7: All of these themes together
      Please make a selection!
      1
      One last question: What size do you want?
      Please choose from the following:
       1: One picture
       2: 7*7 collection of pictures
     ```
- The software then starts running making the following print for each epoch: 

    ```Python 
     Epoch 1, gen loss = 2.1588640213012695, disc loss = 1.230824589729309
    ```

- The generated artwork gets saved in the Output folder, under the corresponding theme folder. If the user generated using the one picture option, the picture might be too small however we have the Img_Resizing.py program to tackle that as the following: 

    ```Python 
     python Img_Resize.py 
    ```
- The user then will get the following output to select the folder of images that needs to be resized: 

    ```Python 
     Hello! Congrats You Got Some Nice Generated Images!!
     Which folder you want to resize?
     Please choose from the following:
     1: Demons
     2: Hell
     3: Psychedelic
     4: Purgatory
     5: Dark
     6: Depression
     7: All themes 
    ```
- Once the user makes a selection of which theme needs to be resized, the resized images gets saved under the Resized folder, under the corresponding theme. 


## Folder Structure 

This repo consists of the following folders structure: 
- RH GAN: Main GAN software folder 
    - Data_NPY_Files: Input Datasets as NPY files for all themes supported by the software. This folder should be downloaded from the DropBox or the Mega link. 
        - dark.npy
        - demons.npy 
        - depression.npy
        - hell.npy
        - purgatory.npy
        - psychedelic.npy
        - allThemes.npy
    - Output: The output folders where the outputs are saved for each theme 
        - dark_output 
        - depression_output
        - demons_output 
        - hell_output 
        - purgatory_output 
        - psychedelic_output 
        - allThemes_output 
    - Resized: The resized output where the resized outputs are saved after running the Img_Resize.npy on an Output folder. 
        - dark_resized
        - depression_resized
        - demons_resized 
        - hell_resized 
        - purgatory_resized
        - psychedelic_resized 
        - allThemes_resized 
    - GAN.py: The main software that generates artwork after collecting user input for the theme and size selection.  
    - Img_Resize.py: The resizing program that resizes output images. 
- Supplemental Code: Supplemental programs that were used to scrape, pre-process, pre-load and save images as NPY files. 
    - Scraped_Images: The folder that contains the file that resize all scraped images and save them to an NPY file. 
        - NPY_resize.py 
    - Scraping.py: The scrapping program that scrapes images from Google Images. 
    - Rotate.py: The Rotating program that rotates the scrapped images to have a larger dataset. 
- Environement.yaml: This file will allow you to load the same conda environement with the required packages that we used to develop and run this project.
 

## Supplemental Documents and Resources Used:
- https://www.kaggle.com/code/isaklarsson/gan-art-generator/notebook
- https://towardsdatascience.com/generating-modern-arts-using-generative-adversarial-network-gan-on-spell-39f67f83c7b4
- https://ladvien.com/scraping-internet-for-magic-symbols/
- https://stackoverflow.com/questions/56368107/rotation-of-images
