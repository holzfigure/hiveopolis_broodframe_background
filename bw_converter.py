### Script for turning a bunch of images in a folder to greyscale ###
## (This reduces the file size drastically, for example for raspberry pi NoIr images that are grey scale anyways)

from PIL import Image
import os, sys

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#Path of folder with existing files
path = "F://300_Bilder"
#New path for smaller files (can be the same, then images are overwritten)
np = "F://300_Bilder_smoll"

#check if new folder exists, if not create it
check_dir(np)

#Iterate over every file in folder
for filename in os.listdir(path):
    fn = "{}/{}".format(path, filename)
    nfn = "{}/{}".format(np, filename)
    img = Image.open(fn).convert('L')
    img.save(nfn, quality=95)

