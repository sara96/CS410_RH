################################################## IMPORTS #######################################################
import os
import subprocess
from PIL import Image

# Keras & Tensorflow libraries 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.layers import Dropout, LeakyReLU, BatchNormalization
from keras.layers import Activation, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Reshape
from matplotlib import pyplot
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Numpy libraries
import numpy
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from numpy import zeros
from numpy import ones
from numpy import asarray

#Torchvision for fast and easy loading and resizing
import torchvision
import torchvision.transforms as transforms

# Terminal Output Coloring 
import colorama
from colorama import Fore, Back, Style 
################################################## END OF IMPORTS ####################################################


################################################## GLOBAL VARIABLES ##################################################
# FOR TRAINING CONFIGURATION 
EPOCHS = 251
BATCH_SIZE = 25
SEED_SIZE = 100
# FOR IMAGES SIZE
WIDTH = 64
HEIGHT = 64
# FOR THE OUTPUT 
MARGIN = 2 
CHANNELS = 3
############################################# END OF GLOBAL VARIABLES ##################################################


################################################## CLIENT SELECTION ####################################################
print(" \n\n\nHello! Welcome To Your Favorite Art Generator! \nGet Ready To Build Something Amazing With RHGAN!")
print("Here is a list of the themes we have: ")
print(Fore.RED + " 1: Demons", Fore.MAGENTA + "\n 2: Hell", Fore.YELLOW + "\n 3: Psychedelic", Fore.LIGHTRED_EX + "\n 4: Purgatory", Fore.LIGHTBLACK_EX + "\n 5: Dark", Fore.LIGHTBLUE_EX + "\n 6: Depression", Fore.WHITE + "\n 7: All of these themes together")
print(Fore.GREEN + "Please make a selection!" + Style.RESET_ALL)
Themeselection = input()
print("One last question: What size do you want?")
print("Please choose from the following: ")
print(Fore.RED + " 1: One picture", Fore.MAGENTA + "\n 2: 7*7 collection of pictures", Fore.YELLOW + Style.RESET_ALL)
sizeSelection = input()
################################################## END OF CLIENT SELECTION ##############################################


################################################## DATA LOADING ########################################################
def dataLoader(argument):
    switcher = {
        1: "demons.npy",
        2: "hell.npy",
        3: "psychedelic.npy",
        4: "purgatory.npy",
        5: "dark.npy",
        6: "depression.npy",
        7: "allThemes.npy"
    }
    return switcher.get(argument, "Wrong selection, please try again")

argument = int(Themeselection)
dataSetRequested = dataLoader(argument)
path = dataSetRequested
dataset = np.load("Data_NPY_Files" + '/' + dataSetRequested)
training_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(9000).batch(BATCH_SIZE)

def sizeLoader(argument):
    switcher = {
        1: "1",
        2: "7"
    }
    return switcher.get(argument, "Wrong selection, please try again")
argument = int(sizeSelection)
GENERATED_ROWS = int(sizeLoader(argument))
GENERATED_COLS = int(sizeLoader(argument))
################################################## END OF DATA LOADING #################################################


################################################## GENERATOR MODEL #####################################################
def Generator(seed_size, channels):
    
    model = Sequential()

    model.add(Dense(64*64,activation="relu",input_dim=seed_size)) 
    model.add(Reshape((4,4,256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   

    model.add(UpSampling2D(size = (2,2))) 
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    
    model.add(UpSampling2D(size=(2,2)))
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))
    
    return model

################################################## END OF GENERATOR MODEL ################################################


################################################## DISCRIMNATOR MODEL ####################################################
def Discrimnator(imgShape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=imgShape, 
                     padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model
################################################## END OFDISCRIMNATOR MODEL #################################################


################################################## LOSS FUNCTIONS ###########################################################
CE = tf.keras.losses.BinaryCrossentropy()

# DISCRIMNATOR LOSS 
def DiscLoss(real, fake):
    XReal = CE(tf.ones_like(real), real)
    YFake = CE(tf.zeros_like(fake), fake)
    total = XReal + YFake
    return total

# GENERATOR LOSS
def GenLoss(fake):
    return CE(tf.ones_like(fake), fake)
################################################## END OF LOSS FUNCTIONS #####################################################


##################################################### OPTIMIZATION ###########################################################
GenOptimizer = tf.keras.optimizers.Adam(1.2e-4, 0.5)
DiscOptimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5)
##################################################### END OF OPTIMIZATION #####################################################


################################################## BUILDING MODELS ##########################################################
# BUILDING GENERATOR 
BuildGen = Generator(SEED_SIZE,CHANNELS)
imgShape = (HEIGHT, HEIGHT, CHANNELS)

# BUILDING DISCRIMNATOR 
BuildDiscriminator = Discrimnator(imgShape)
################################################## END OF BUILDING MODELS ###################################################


##################################################### TRAINING STEP ###########################################################
@tf.function
def training(images):
    seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])
    
    with tf.GradientTape() as genT, tf.GradientTape() as discT:
        gen_img = BuildGen(seed, training = True)
        
        real = BuildDiscriminator(images, training=True)
        fake = BuildDiscriminator(gen_img, training=True)
        
        gen_loss = GenLoss(fake)
        disc_loss = DiscLoss(real, fake)
        
        
        gradients_of_generator = genT.gradient(
            gen_loss,
            BuildGen.trainable_variables
        )
        gradients_of_discriminator = discT.gradient(
            disc_loss, 
            BuildDiscriminator.trainable_variables
        )
        
        GenOptimizer.apply_gradients(zip(gradients_of_generator,
                                                BuildGen.trainable_variables))
        
        DiscOptimizer.apply_gradients(zip(gradients_of_discriminator, 
                                                    BuildDiscriminator.trainable_variables))
        
        return gen_loss, disc_loss

##################################################### END OF TRAINING STEPS #########@#######################################


##################################################### OUTPUT SAVING #########################################################
def save(cnt, noise):
    #Define the "base" of the saved image as a big black canvas
    image_array = np.full(( 
       MARGIN + (GENERATED_ROWS * (WIDTH + MARGIN)), 
       MARGIN + (GENERATED_COLS * (HEIGHT + MARGIN)), 3), 
        0, dtype=np.uint8)
    
    gen_img =  BuildGen.predict(noise)
    
    image_count = 0
    for row in range(GENERATED_ROWS):
        for col in range(GENERATED_COLS):
            r = row * (WIDTH + 2) + MARGIN
            c = col * (HEIGHT + 2) + MARGIN
            image_array[r:r+WIDTH , c:c+HEIGHT] = gen_img[image_count] * 127.5 + 127.5
            image_count += 1

    base, ext = os.path.splitext(dataSetRequested)     
    output_path  = 'Output/' + path[:-4] + "_output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path,f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
##################################################### END OF OUTPUT SAVING ##################################################


##################################################### TRAINING ##############################################################
def train(dataset, epochs):
    # Use a fixed seed for the saved images so we can watch their development
    fixed_seed = np.random.normal(0, 1, (GENERATED_ROWS * GENERATED_COLS, SEED_SIZE))
    
    for epoch in range(epochs):
        gen_loss_list = []
        disc_loss_list = []
        
        for image_batch in dataset:
            t = training(image_batch)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])
            
        g_loss = sum(gen_loss_list) / len(gen_loss_list) #calculate losses
        d_loss = sum(disc_loss_list) / len(disc_loss_list)
        
        print(f'Epoch {epoch+1}, gen loss = {g_loss}, disc loss = {d_loss}')
        
        save(epoch, fixed_seed)
##################################################### END OF TRAINING ######################################################        


##################################################### MAIN FUNCTION ########################################################
if __name__ == "__main__":
    train(training_dataset, EPOCHS)
##################################################### END OF PROGRAM ########################################################
      
