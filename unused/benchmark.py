########################################
########################################
# Python script to test various algorithms
# as benchmark for quality and speed
# to remove moving bees from broodframe
########################################
########################################

# Use open cv version 3 for best support
# if you other version installed and no enviroments set up you must first uninstall
# the other open cv versions

# best way is virtual enviroments
# go into project folder and then following commands:
# python3 -m venv env
# source env/bin/activate (mac)
# .\env\Scripts\activate (windows)
# (you can leave the enviroment with deactivate)

# check if activation did work:
# which python

# install python libraries needed:
# pip install opencv-python==3.4.5.20
# pip install pybgs

# Basic libraries
import numpy as np
import os
import glob

# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv
# our algorithm library please refer and citate https://github.com/andrewssobral/bgslibrary
# c++ but with python wrapper, has multiple algorithms which are not standard included in open cv
# does background image generation with the foremask from cv2
import pybgs as bgs
# timer are used for stats, too compare running time of algorithms
import time


print("OpenCV Version: {}".format(cv.__version__))

def is_cv2():
  return check_opencv_version("2.")

def is_cv3():
  return check_opencv_version("3.")

def is_cv4():
  return check_opencv_version("4.")

def check_opencv_version(major):
  return cv.__version__.startswith(major)

# empty array of algorithms
algorithms = []

## bgslibrary algorithms
#algorithms.append(bgs.FrameDifference())
#algorithms.append(bgs.StaticFrameDifference())
#algorithms.append(bgs.WeightedMovingMean())
#algorithms.append(bgs.WeightedMovingVariance())
#algorithms.append(bgs.AdaptiveBackgroundLearning())
#algorithms.append(bgs.AdaptiveSelectiveBackgroundLearning())
algorithms.append(bgs.MixtureOfGaussianV2())
#if is_cv2():
  #algorithms.append(bgs.MixtureOfGaussianV1()) # if opencv 2.x
  #algorithms.append(bgs.GMG()) # if opencv 2.x
#if is_cv3():
  #algorithms.append(bgs.KNN()) # if opencv 3.x
#if is_cv2() or is_cv3():
  #algorithms.append(bgs.DPAdaptiveMedian())
  #algorithms.append(bgs.DPGrimsonGMM())
  #algorithms.append(bgs.DPZivkovicAGMM())
  #algorithms.append(bgs.DPMean())
  #algorithms.append(bgs.DPWrenGA())
  #algorithms.append(bgs.DPPratiMediod())
  #algorithms.append(bgs.DPEigenbackground())
  #algorithms.append(bgs.DPTexture())
  #algorithms.append(bgs.T2FGMM_UM())
  #algorithms.append(bgs.T2FGMM_UV())
  #algorithms.append(bgs.T2FMRF_UM())
  #algorithms.append(bgs.T2FMRF_UV())
  #algorithms.append(bgs.FuzzySugenoIntegral())
  #algorithms.append(bgs.FuzzyChoquetIntegral())
  #algorithms.append(bgs.LBSimpleGaussian())
  #algorithms.append(bgs.LBFuzzyGaussian())
  #algorithms.append(bgs.LBMixtureOfGaussians())
  #algorithms.append(bgs.LBAdaptiveSOM())
  #algorithms.append(bgs.LBFuzzyAdaptiveSOM())
  #algorithms.append(bgs.LBP_MRF())
  #algorithms.append(bgs.MultiLayer())
  #algorithms.append(bgs.PixelBasedAdaptiveSegmenter())
  #algorithms.append(bgs.VuMeter())
  #algorithms.append(bgs.KDE())
  #algorithms.append(bgs.IndependentMultimodal())
  #algorithms.append(bgs.MultiCue())
#algorithms.append(bgs.SigmaDelta())
#algorithms.append(bgs.SuBSENSE())
#algorithms.append(bgs.LOBSTER())
#algorithms.append(bgs.PAWCS())
#algorithms.append(bgs.TwoPoints())
#algorithms.append(bgs.ViBe())
#algorithms.append(bgs.CodeBook())

# path to raw files
path_raw = 'img'

# define minimum of runs before creating the first background
# if is atm not included if(x > min_runs):
min_runs = 300

max_runs = 100

# load all img as array reverse sorted with oldest at beginning
array = sorted(glob.iglob(path_raw + '/*.jpg'), key = os.path.getctime, reverse = True)

print("Starting with total file number: " + str(len(array)))

for algorithm in algorithms:

    print("Starting algorithm: " + algorithm.__class__.__name__)
    # create algorithm folder for finished files
    try:
        os.mkdir(algorithm.__class__.__name__)
    except:
        print("Folder is already created")

    # start time for loop of algorithm
    start_time = time.time()

    # loop x times as files in our folder
    for x in range(0, len(array)):

        # we can loop now through our array of images
        img_path = array[x]

        print("Loaded oldest image: ", img_path)

        # read file into open cv and apply to algorithm to generate background model
        img = cv.imread(img_path)
        img_output = algorithm.apply(img)
        img_bgmodel = algorithm.getBackgroundModel()

        # TODO set minimum number were to start to save images otherwise the first few hundred are not good
        # dont write background as long we havent reached our minimum learning value
        # if(x > min_runs):
        #print("Write Background")
        # wite into algorithm folder the finished backgroundModels
        img_bg = img_path.replace(path_raw, algorithm.__class__.__name__)
        # print(img_bg)
        cv.imwrite(img_bg, img_bgmodel)
        # if ends if TODO included

        if (x == max_runs):
            break

        print("Runs left: " + str(len(array)-x))

    # end time of algoritm
    end_time = time.time()
    string_time = str(end_time - start_time)

    print("--- %s seconds finished algorithm ---" % string_time)

    text_file = open(algorithm.__class__.__name__+"/time.txt", "w")
    text_file.write("Time (s): %s" % string_time)
    text_file.close()

# END
