#!/usr/bin/env python3
"""This script removes bees from brood frame photos.

Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm.

The class implements the Gaussian mixture model background subtraction
described in:
(1) Z.Zivkovic, Improved adaptive Gausian mixture model for background
  subtraction, International Conference Pattern Recognition, UK, August, 2004,
  The code is very fast and performs also shadow detection.
  Number of Gausssian components is adapted per pixel.
(2) Z.Zivkovic, F. van der Heijden, Efficient Adaptive Density Estimation
  per Image Pixel for the Task of Background Subtraction,
  Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.
  The algorithm similar to the standard Stauffer&Grimson algorithm
  with additional selection of the number of the Gaussian components based on:
  Z.Zivkovic, F.van der Heijden, Recursive unsupervised learning of finite
  mixture models, IEEE Trans. on Pattern Analysis and Machine Intelligence,
  vol.26, no.5, pages 651-656, 2004.
"""

# Basic libraries
import os
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv


# SETTINGS
# Path to raw files and output folder of generated images
# PATH_RAW = Path.cwd() / 'img'
# PATH_OUT = Path.cwd() / 'out'
PATH_RAW = Path("/media/holzfigure/Data/NAS/NAS_incoming_data")
PATH_OUT = Path("/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis")

# https://docs.opencv.org/master/d7/df6/
# classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
LEARNING_RATE = -1
# Max runs if we want to limit generation, False or 0 for no max runs
MAX_RUNS = 0
# Print modulus, only used for output of text
PRINT_MODULUS = 10
# MOG Settings
HISTORY = 200           # standard 200
SHADOW = False          # just returns detected shadows, we dont need it
VAR_THRESHOLD = 16       # standard 16
# Use sharpen algorithm, takes a lot of time and
# needs opencv4 (pip3 install opencv-python)
SHARPEN = True
# Adjust Gamma
ADJUST_GAMMA = True
# END SETTINGS

# print("###########  SETTINGS  ##################")
# print("###########  Path raw: {}".format(path_raw))
# print("###########  Path output: {}".format(path_out))
# print("###########  Print modulus: {}".format(print_modulus))
# print("###########  Max runs: {}".format(max_runs))
# print("###########  Learning rate: {}".format(learning_rate))
# print("###########  History: {}".format(history))
# print("###########  Shadow: {}".format(shadow))
# print("###########  VarThreshold: {}".format(var_threshold))
# print("###########  Sharpen: {}".format(sharpen))
# print("###########  Adjust Gamma: {}".format(adjust_gamma))


def adjust_gamma(image, ltable):
    """Apply gamma correction using the lookup table."""
    return cv.LUT(image, ltable)


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0,
                 amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask.

    https://en.wikipedia.org/wiki/Unsharp_masking
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm"""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        # OpenCV4 function copyTo
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def main(
    path_raw=PATH_RAW,
    path_out=PATH_OUT,
    learning_rate=LEARNING_RATE,
    max_runs=MAX_RUNS,
    print_modulus=PRINT_MODULUS,
    history=HISTORY,
    shadow=SHADOW,
    var_threshold=VAR_THRESHOLD,
    sharpen=SHARPEN,
    adjust_gamma=ADJUST_GAMMA,
):

    # Check whether input path is available
    # if not os.path.isdir(path_raw):
    if not path_raw.is_dir():
        print(f"Error: Missing folder {path_raw}, now leaving..")
        # exit()
        # sys.exit(1)
        raise SystemExit(1)

    # Create output folder
    # try:
    # if not os.path.isdir(path_out):
    if not path_out.is_dir():
        path_out.mkdir()  # parents=True)
    # except:
    else:
        print(f"Folder '{path_out}' already exists")
        i = input(f"Clear files in folder '{path_out}' (y/N)?")
        if i == "y":
            # Empty output folder
            for the_file in os.listdir(path_out):
                file_path = os.path.join(path_out, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
            print("Files deleted")

    print(
        "======= SETTINGS =======\n"
        f"Path raw:      {path_raw}\n"
        f"Path output:   {path_out}\n"
        f"Print modulus: {print_modulus}\n"
        f"Max runs:      {max_runs}\n"
        f"Learning rate: {learning_rate}\n"
        f"History:       {history}\n"
        f"Shadow:        {shadow}\n"
        f"var_threshold: {var_threshold}\n"
        f"Sharpen:       {sharpen}\n"
        f"Adjust Gamma:  {adjust_gamma}\n"
    )

    # Load CV BackgroundSubtractorMOG2 with settings
    # https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    # https://docs.opencv.org/master/d6/d17/group__cudabgsegm.html
    # Attributes found inside Class
    # 'setBackgroundRatio', 'setComplexityReductionThreshold',
    # 'setDetectShadows', 'setHistory', 'setNMixtures',
    # 'setShadowThreshold', 'setShadowValue', 'setVarInit', 'setVarMax',
    # 'setVarMin', 'setVarThreshold', 'setVarThresholdGen'
    mog = cv.createBackgroundSubtractorMOG2()
    mog.setDetectShadows(shadow)
    mog.setHistory(history)
    mog.setVarThreshold(var_threshold)

    # Process all folders
    folders = sorted(path_raw.glob("Photos_of_Pi*"))
    print(f"Number of folders: {len(folders)}")

    for folder in folders:
        # mtime = folder.stat().st_mtime

        # Parse time from foldernames
        # Folder name e.g.: Photos_of_Pi1_1_9_2019
        # t_str = folder.name.split("Photos_of_Pi")[-1][2:]  # heating!!
        t_str = "_".join(folder.name.split("_")[-3:])
        pdt = datetime.strptime(t_str, '%d_%m_%Y')

        # Parse time from filenames
        pt = datetime.strptime(tstr, '%d_%m_%H_%M_%S')
        # TODO: set year to 2019



    # TODO: Find pathlib way of doing this: key=os.path.getmtime
    #       p.stat().st_mtime
    # load all img as array reverse sorted with oldest at beginning
    # Image name e.g.: pi1_hive1broodn_1_8_0_0_3.jpg

    # array = sorted(glob.iglob(path_raw + '/*.jpg'),
    #                key=os.path.getmtime, reverse=True)
    # Try this:
    # array = sorted(path_raw.rglob("*.jpg"),
    #                key=os.path.getmtime, reverse=True)



    print("#############################")
    print("Starting with total file number: " + str(len(array)))
    print("#############################")

    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    # Somehow I found the value of `gamma=1.2` to be the best in my case
    inv_gamma = 1.0 / 1.2
    ltable = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # loop x times as files in our folder
    for x in range(len(array)):

        # We can loop now through our array of images
        img_path = array[x]

        # Read file with OpenCV
        img = cv.imread(img_path)

        # Preprocessing ######
        # Change image to grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Equalize grayscale image
        # img = cv.equalizeHist(img)
        # Add gaussian to remove noise
        # img = cv.GaussianBlur(img, (1, 1), 0)
        # img = cv.medianBlur(img, 1)
        # img = cv.GaussianBlur(img, (7, 7), 1.5)
        # END Preprocessing ######

        # Apply algorithm to generate background model
        img_output = mog.apply(img, learning_rate)
        print(f"img_out: {img_output}")
        # Threshold for foreground mask, we don't use the
        # foreground mask so we dont need it?
        # img_output = cv.threshold(img_output, 10, 255, cv.THRESH_BINARY);

        # Get background image
        img_bgmodel = mog.getBackgroundImage()

        # Postprocessing ######

        # Sharpen the image
        if sharpen:
            img_bgmodel = unsharp_mask(img_bgmodel)

        # Adjust gamma if there is light change
        if adjust_gamma:
            img = adjust_gamma(img, ltable)

        # Change image to grayscale
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Equalize grayscale image
        # img = cv.equalizeHist(img)
        # Add gaussian to remove noise
        # img = cv.GaussianBlur(img, (1, 1), 0)
        # img = cv.medianBlur(img, 1)
        # img = cv.GaussianBlur(img, (7, 7), 1.5)

        # img_bgmodel = cv.equalizeHist(img_bgmodel)
        # END Preprocessing ######

        # Write finished backgroundModels
        img_bg = img_path.replace(path_raw, path_out)
        cv.imwrite(img_bg, img_bgmodel)

        # Break if max runs is defined and reached
        if max_runs > 0:
            if x == max_runs:
                break

        if (x % print_modulus) == 0:
            print(f"Current image: {img_path}\nRuns left: {len(array)-x}")

    # END


if __name__ == "__main__":
    main()  # (args)
