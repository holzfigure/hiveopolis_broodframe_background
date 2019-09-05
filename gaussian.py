########################################
########################################
# This script removes bees from brood frames
# of a series of pictures with the Gaussian Logic
########################################
########################################
########################################
# Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm.
#
# The class implements the Gaussian mixture model background subtraction described in:
# (1) Z.Zivkovic, Improved adaptive Gausian mixture model for background subtraction, International Conference Pattern Recognition, UK, August, 2004,
# The code is very fast and performs also shadow detection. Number of Gausssian components is adapted per pixel.
#
# (2) Z.Zivkovic, F. van der Heijden, Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction,
# Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.
# The algorithm similar to the standard Stauffer&Grimson algorithm with additional selection of the number of the Gaussian components based on:
# Z.Zivkovic, F.van der Heijden, Recursive unsupervised learning of finite mixture models, IEEE Trans. on Pattern Analysis and Machine Intelligence,
# vol.26, no.5, pages 651-656, 2004.
########################################
########################################

# Basic libraries
import numpy as np
import os
import glob
# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv

############## SETTINGS #######################
# path to raw files and output folder of generated images
path_raw = 'img'
path_output = 'out'
# https://docs.opencv.org/master/d7/df6/classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
learning_rate = -1
# max runs if we want to limit generation, False or 0 for no max runs
max_runs = 0
# print modulus, only used for output of text
print_modulus = 10
#### MOG Settings
history = 200           # standard 200
shadow = False          # just returns detected shadows, we dont need it
VarThreshold = 16       # standard 16
# use sharpen algorithm, takes a lot of time and needs opencv4 (pip3 install opencv-python)
sharpen = True
# use to adjust Gamma
adjustGamma = True
############## END SETTINGS #######################

print("###########  SETTINGS  ##################")
print("###########  Path raw: {}".format(path_raw))
print("###########  Path output: {}".format(path_output))
print("###########  Print modulus: {}".format(print_modulus))
print("###########  Max runs: {}".format(max_runs))
print("###########  Learning rate: {}".format(learning_rate))
print("###########  History: {}".format(history))
print("###########  Shadow: {}".format(shadow))
print("###########  VarThreshold: {}".format(VarThreshold))
print("###########  Sharpen: {}".format(sharpen))
print("###########  Adjust Gamma: {}".format(adjustGamma))

# load CV BackgroundSubtractorMOG2 with settings
# https://docs.opencv.org/master/d6/d17/group__cudabgsegm.html
# attribues found inside Class
# 'setBackgroundRatio', 'setComplexityReductionThreshold', 'setDetectShadows', 'setHistory', 'setNMixtures', 'setShadowThreshold', 'setShadowValue', 'setVarInit', 'setVarMax', 'setVarMin', 'setVarThreshold', 'setVarThresholdGen'
mog = cv.createBackgroundSubtractorMOG2();
mog.setDetectShadows(shadow);
mog.setHistory(history);
mog.setVarThreshold(VarThreshold)


if(os.path.isdir(path_raw) == False):
    print("####### Missng {} folder ###########".format(path_raw))
    exit()

# load all img as array reverse sorted with oldest at beginning
array = sorted(glob.iglob(path_raw + '/*.jpg'), key = os.path.getmtime, reverse = True)

# create output folder if not exists
try:
    os.mkdir(path_output)
except:
    print("{} Folder already exists".format(path_output))
    i = input("Clear files in {} folder (y/n)?".format(path_output))
    if (i == "y"):
        # empty output folder
        for the_file in os.listdir(path_output):
            file_path = os.path.join(path_output, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print("Files deleted")

print("#############################")
print("Starting with total file number: " + str(len(array)))
print("#############################")



# build a lookup table mapping the pixel values [0, 255] to
# their adjusted gamma values
invGamma = 1.0 / 1.2
ltable = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

# Somehow I found the value of `gamma=1.2` to be the best in my case
def adjust_gamma(image, ltable):
    # apply gamma correction using the lookup table
    return cv.LUT(image, ltable)

# Return a sharpened version of the image, using an unsharp mask.
# https://en.wikipedia.org/wiki/Unsharp_masking
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
def unsharp_mask(image, kernel_size= (5, 5), sigma = 1.0, amount = 1.0, threshold = 0):
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        # opencv4 function copyTo
        np.copyto(sharpened, image, where = low_contrast_mask)
    return sharpened

# loop x times as files in our folder
for x in range(0, len(array)):

    # we can loop now through our array of images
    img_path = array[x]

    # read file into open cv and apply to algorithm to generate background model
    img = cv.imread(img_path)

    #### Preprocessing ######
    # change image to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # equalize grayscale image
    # img = cv.equalizeHist(img)
    # add gaussian to remove noise
    #img = cv.GaussianBlur(img, (1, 1), 0)
    #img = cv.medianBlur(img, 1)
    #img = cv.GaussianBlur(img, (7, 7), 1.5)
    #### END Preprocessing ######

    img_output = mog.apply(img, learning_rate);

    # threshold for foreground mask, we don't use the foreground mask so we dont need it?
    #img_output = cv.threshold(img_output, 10, 255, cv.THRESH_BINARY);

    # get background image
    img_bgmodel = mog.getBackgroundImage();

    #### Postprocessing ######

    # sharpens the image
    if(sharpen == True):
        img_bgmodel = unsharp_mask(img_bgmodel)

    # adjust gamma if there is light change
    if (adjustGamma == True):
        img = adjust_gamma(img, ltable)

    # change image to grayscale
    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # equalize grayscale image
    # img = cv.equalizeHist(img)
    # add gaussian to remove noise
    #img = cv.GaussianBlur(img, (1, 1), 0)
    #img = cv.medianBlur(img, 1)
    #img = cv.GaussianBlur(img, (7, 7), 1.5)

    #img_bgmodel = cv.equalizeHist(img_bgmodel)
    #### END Preprocessing ######

    # wite finished backgroundModels
    img_bg = img_path.replace(path_raw, path_output)
    cv.imwrite(img_bg, img_bgmodel)

    # break if max runs is defined and reached
    if(max_runs > 0):
        if (x == max_runs):
            break

    if (x % print_modulus == 0):
        print("Current image: ", img_path + "\n Runs left: " + str(len(array)-x))

# END
