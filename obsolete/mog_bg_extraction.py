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

# Standard libraries
# import os
# import glob
# import logging
import argparse
# import platform
from pathlib import Path
from datetime import datetime

# Third-party libraries
import pytz
import pandas as pd
import numpy as np
# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv

# BACKGROUND-EXTRACTION SETTINGS
# https://docs.opencv.org/master/d7/df6/
# classcv_1_1BackgroundSubtractor.html#aa735e76f7069b3fa9c3f32395f9ccd21
LEARNING_RATE = -1

# MOG Settings
HISTORY = 200           # standard 200
VAR_THRESHOLD = 16       # standard 16
# Use sharpen algorithm, takes a lot of time and
# needs opencv4 (pip3 install opencv-python)
SHARPEN = True
# Adjust Gamma
ADJUST_GAMMA = True
# END SETTINGS
SHADOW = False          # just returns detected shadows, we dont need it
# Max runs if we want to limit generation, False or 0 for no max runs
MAX_RUNS = 0

# Build a lookup table mapping the pixel values [0, 255] to
# their adjusted gamma values
# Somehow I found the value of `gamma=1.2` to be the best in my case
INV_GAMMA = 1.0 / 1.2
GAMMA_LOOKUP_TABLE = np.array([
    ((i / 255.0) ** INV_GAMMA) * 255
    for i in np.arange(0, 256)]).astype("uint8")

# Print modulus, only used for output of text
PRINT_MODULUS = 10


# argument parsing
parser = argparse.ArgumentParser(
    description=("Extract the broodnest from colony photos."))
parser.add_argument("-d", "--debug", action="store_true",
                    help="debug mode")
parser.add_argument("-i", "--interactive", action="store_true",
                    help="popup dialog to select files or folders")
# parser.add_argument("-p", "--plot", action="store_true",
#                     help="make various plots of the data")
parser.add_argument("-r", "--learningrate", type=int,
                    default=LEARNING_RATE,
                    help=(
                        "set learning rate for "
                        "background extraction algorithm, "
                        "negative numbers auto-infer "
                        "(default: %(default)s)"
                    ))
parser.add_argument("-n", "--history", type=int,
                    default=HISTORY,
                    help=(
                        "number of photos used for background extraction "
                        "(default: %(default)s)"
                    ))
parser.add_argument("-v", "--varthreshold", type=int,
                    default=VAR_THRESHOLD,
                    help=(
                        "??? 'standard 16' ??? "
                        "(default: %(default)s)"
                    ))
parser.add_argument("-s", "--sharpen", action="store_true",
                    help="use 'sharpen' algorithm")
parser.add_argument("-g", "--adjustgamma", action="store_true",
                    help="adjust gamma...")
parser.add_argument("--shadow", action="store_false",
                    help="use 'shadow-detection' algorithm")
parser.add_argument("-m", "--maxruns", type=int,
                    default=MAX_RUNS,
                    help=(
                        "??? "
                        "'Max runs if we want to limit generation, "
                        "False or 0 for no max runs' ??? "
                        "(default: %(default)s)"
                    ))


def extract_background(
        df,  # pandas series of pathlib Paths to the photos
        path_raw,
        path_out,
        bg_folder=BG_OUTFOLDER,
        time_fmt=TIME_FMT,
        day_fmt=DAY_FMT,
        prefix=OUTIMG_PREFIX,
        print_modulus=PRINT_MODULUS,
        learning_rate=ARGS.learningrate,
        max_runs=ARGS.maxruns,
        history=ARGS.history,
        shadow=ARGS.shadow,
        var_threshold=ARGS.varthreshold,
        sharpen=ARGS.sharpen,           # just to print settings
        adjust_gamma=ARGS.adjustgamma,  # just to print settings
        args=ARGS,
):
    """Run Gaussian stuff using code from Hannes Oberreiter.

    Assuming all files in df are from the same Hive, RPi and DAY!
    """
    # Parse general info
    n_files = len(df)
    hive = df.hive[-1]
    rpi = df.rpi[-1]
    in_folder = path_raw / df.path[0].parent

    # Create output folder
    path_out = path_out / bg_folder / f"hive{hive}/rpi{rpi}"
    if args.debug:
        # path_out = path_out / bg_folder / f"hive{hive}/rpi{rpi}/{out_folder}"
        day_str = df.time[-1].strftime(day_fmt)
        out_folder = f"hive{hive}_rpi{rpi}_{day_str}"
        path_out = path_out / out_folder

    if not path_out.is_dir():
        path_out.mkdir(parents=True)
        print(f"Created folder '{path_out}'")

    # Fix file prefix
    file_prefix = f"{prefix}_hive{hive}_rpi{rpi}"

    print(f"Received {n_files} in '{in_folder}', "
          f"exporting to '{path_out}'")

    print(
        "\n======= MOG SETTINGS =======\n"
        # f"Path raw:      {in_folder}\n"
        # f"Path output:   {path_out}\n"
        f"Print modulus: {print_modulus}\n"
        f"Max runs:      {max_runs}\n"
        f"Learning rate: {learning_rate}\n"
        f"History:       {history}\n"
        f"Shadow:        {shadow}\n"
        f"var_threshold: {var_threshold}\n"
        f"Sharpen:       {sharpen}\n"
        f"Adjust Gamma:  {adjust_gamma}\n"
    )

    # # Load CV BackgroundSubtractorMOG2 with settings
    # # https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    # # https://docs.opencv.org/master/d6/d17/group__cudabgsegm.html
    # # Attributes found inside Class
    # # 'setBackgroundRatio', 'setComplexityReductionThreshold',
    # # 'setDetectShadows', 'setHistory', 'setNMixtures',
    # # 'setShadowThreshold', 'setShadowValue', 'setVarInit', 'setVarMax',
    # # 'setVarMin', 'setVarThreshold', 'setVarThresholdGen'
    mog = cv.createBackgroundSubtractorMOG2()
    mog.setDetectShadows(shadow)
    mog.setHistory(history)
    mog.setVarThreshold(var_threshold)
    #
    # # TODO: Find pathlib way of doing this: key=os.path.getmtime
    # #       p.stat().st_mtime
    # # load all img as array reverse sorted with oldest at beginning
    # # Image name e.g.: pi1_hive1broodn_1_8_0_0_3.jpg
    #
    # # array = sorted(glob.iglob(path_raw + '/*.jpg'),
    # #                key=os.path.getmtime, reverse=True)
    # # Try this:
    # # array = sorted(path_raw.rglob("*.jpg"),
    # #                key=os.path.getmtime, reverse=True)

    # # Build a lookup table mapping the pixel values [0, 255] to
    # # their adjusted gamma values
    # # Somehow I found the value of `gamma=1.2` to be the best in my case
    # inv_gamma = 1.0 / 1.2
    # ltable = np.array([
    #     ((i / 255.0) ** inv_gamma) * 255
    #     for i in np.arange(0, 256)]).astype("uint8")

    # Iterate over all files
    for x in range(n_files):

        # We can loop now through our array of images
        img_path = path_raw / df.path[x]

        # Read file with OpenCV
        img = cv.imread(str(img_path))

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
        # img_output = mog.apply(img, learning_rate)
        # # logging.debug(f"img_out: {img_output}")
        # # Threshold for foreground mask, we don't use the
        # # foreground mask so we dont need it?
        # # img_output = cv.threshold(img_output, 10, 255, cv.THRESH_BINARY);
        mog.apply(img, learning_rate)

        if args.debug:
            filepath = make_filename(path_out, file_prefix, df.index[x])
            export_background(mog.getBackgroundImage(), filepath)

        # Break if max runs is defined and reached
        if max_runs > 0:
            if x == max_runs:
                break

        if x % print_modulus == 0:
            print(f"Current image: {img_path}\n"
                  f"Runs left: {n_files - x}")

    print(f"Iterated over all files in '{in_folder}'")
    filepath = make_filename(path_out, file_prefix, df.index[x])
    export_background(mog.getBackgroundImage(), filepath)

    return None


def main(file_pattern=INFILE_PATTERN, args=ARGS):
    """Extract the background from large amounts of broodnest photos."""
    # Initialize IO-directories and setup logging
    path_in, path_raw, path_out = initialize_io()

    # Get Paths to all CSV-files
    csv_list = sorted(path_in.rglob(file_pattern))
    print(f"Found {len(csv_list)} files "
          f"matching pattern '{file_pattern}' "
          f"in '{path_in}'.")

    for csv_path in csv_list:
        logging.info(f"Reading '{csv_path.name}'")
        # # Works for parsing my time_fmt:
        # df = pd.read_csv(csv_path, index_col="time", parse_dates=True,
        #                  date_parser=my_date_parser)
        # Works only with the default pandas time format:
        df = pd.read_csv(csv_path, index_col="time", parse_dates=True,
                         converters={"path": my_path_parser})
        logging.debug(f"Read in dataframe sized {df.shape}.")

        # Now you don't really need all the fancy CSV-parsing magic..
        # Call the Gaussian Action xaggly hewe Oida!
        bg_path = extract_background(df, path_raw, path_out)

    logging.info("Done.")


if __name__ == "__main__":
    main()  # (args)
