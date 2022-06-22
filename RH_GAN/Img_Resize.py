################################################## IMPORTS #######################################################
from email.mime import image
from importlib.resources import path
import os
from unittest.mock import patch
import numpy as np  # to resize array
from PIL import Image  # to resize iamges
import colorama
from colorama import Fore, Back, Style
################################################## END OF IMPORTS ####################################################

###################################################### USER INPUT ####################################################
print("Hello! Congrats You Got Some Nice Generated Images!!")
print("Which folder you want to resize? \n Please choose from the following: ")
print(Fore.RED + " 1: Demons", Fore.MAGENTA + "\n 2: Hell", Fore.YELLOW + "\n 3: Psychedelic", Fore.LIGHTRED_EX + "\n 4: Purgatory",
      Fore.LIGHTBLACK_EX + "\n 5: Dark", Fore.LIGHTBLUE_EX + "\n 6: Depression", Fore.WHITE + "\n 7: All themes" + Style.RESET_ALL)
selection = input()


def FolderToResize(argument):
    switcher = {
        1: "demons",
        2: "hell",
        3: "psychedelic",
        4: "purgatory",
        5: "dark",
        6: "depression",
        7: "allThemes"
    }
    return switcher.get(argument, "Wrong selection, please try again")


argument = int(selection)
imagesDirectory = FolderToResize(argument)

################################################## END OF USER INPUT #################################################


################################################## GLOBAL VARIABLES ##################################################
IMAGE_DIR = "Output/" + imagesDirectory + "_output/"
IMAGE_SIZE = 256
IMAGE_CHANNELS = 3
############################################# END OF GLOBAL VARIABLES ################################################


##################################################### RESIZE FUNCTION ################################################
def resize():
    for image in os.listdir(IMAGE_DIR):
        if os.path.isfile(IMAGE_DIR + image):
            im = Image.open(IMAGE_DIR + image)
            f, e = os.path.split(IMAGE_DIR + image)
            fileNameWithExtension = os.path.splitext(e)
            fileName = os.path.basename(fileNameWithExtension[0])
            imResize = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            imResize.save("Resized/" + imagesDirectory + "_resized/" +
                          str(fileName) + '_resized.jpg', 'JPEG', quality=90)
##################################################### END OF RESIZE FUNCTION ############################################


##################################################### MAIN FUNCTION #####################################################
resize()
##################################################### END OF PROGRAM #####################################################
