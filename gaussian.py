# Basic libraries
import numpy as np
import os
import glob
# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv


# path to raw files and output folder of generated images
path_raw = 'img'
path_output = 'out'

# load CV BackgroundSubtractorMOG2 with settings
# https://docs.opencv.org/master/d6/d17/group__cudabgsegm.html
# attribues found inside Class
# 'setBackgroundRatio', 'setComplexityReductionThreshold', 'setDetectShadows', 'setHistory', 'setNMixtures', 'setShadowThreshold', 'setShadowValue', 'setVarInit', 'setVarMax', 'setVarMin', 'setVarThreshold', 'setVarThresholdGen'
mog = cv.createBackgroundSubtractorMOG2();
mog.setDetectShadows(False);
mog.setHistory(200);
mog.setVarThreshold(16)

# https://docs.opencv.org/master/d7/df6/classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
learning_rate = -1

# threshold how to build it into the code?
#threshold = 0.05
#cv.threshold(img_foreground, img_foreground, threshold, 255, cv.THRESH_BINARY);

# max runs if we want to limit generation
max_runs = 500000

if(os.path.isdir(path_raw) == False):
    print("####### Missng {} folder ###########".format(path_raw))
    exit()

# load all img as array reverse sorted with oldest at beginning
array = sorted(glob.iglob(path_raw + '/*.jpg'), key = os.path.getctime, reverse = True)

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

print("Starting with total file number: " + str(len(array)))

# loop x times as files in our folder
for x in range(0, len(array)):

    # we can loop now through our array of images
    img_path = array[x]

    print("Load image: ", img_path)

    # read file into open cv and apply to algorithm to generate background model
    img = cv.imread(img_path)
    img_output = mog.apply(img, learning_rate);
    img_bgmodel = mog.getBackgroundImage();

    # wite into algorithm folder the finished backgroundModels
    img_bg = img_path.replace(path_raw, path_output)
    # print(img_bg)
    cv.imwrite(img_bg, img_bgmodel)
    # if ends if TODO included

    if (x == max_runs):
        break

    print("Runs left: " + str(len(array)-x))


# END
