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
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime

# Third-party libraries
import pytz
import pandas as pd
import numpy as np
# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv

# Own libraries
import iohelp as ioh

# IO-SETTINGS
POSTFIX = None
DEPENDENCIES = [
    Path("iohelp.py"),
    # Path("holzhelp/holzhelp_tk.py"),
    # Path("holzhelp/holzplot.py"),
]
# Path to raw files and output folder of generated images
# PATH_RAW = Path.cwd() / 'img'
# PATH_OUT = Path.cwd() / 'out'
PATH_IN = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/"
    "broodnest_bgs/assemble_paths_191108-utc_default-timefmt/csv"
    # "broodnest_bgs/assemble_paths_191109-utc_my-timefmt/csv"
)
PATH_OUT = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/"
    "broodnest_bgs"
)
# Filename e.g.:  bgx_hive1_rpi2_targ190907-14_190907-140003-utc.csv
INFILE_PATTERN = "bgx_hive*_rpi*_targ*.csv"
# INFOLDER_PATTERN = "Photos_of_Pi*/"
OUTIMG_PREFIX = "bgx"
BG_OUTFOLDER = "imgs"


# TIME-RELATED SETTINGS
# LOCAL_TZ = pytz.timezone("Etc/UTC")
LOCAL_TZ = pytz.timezone("Europe/Vienna")
TIME_FMT = "%y%m%d-%H%M%S-utc"
TIME_TARGET_FMT = "%y%m%d-%H"
# DAY_FMT = "day-%y%m%d"

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
parser.add_argument("-v", "--var-threshold", type=int,
                    default=VAR_THRESHOLD,
                    help=(
                        "??? 'standard 16' ??? "
                        "(default: %(default)s)"
                    ))
# parser.add_argument("-c", "--copy_imgs", type=int,
#                     default=N_IMAGE_COPIES,
#                     help=("save N copies of each plot " +
#                           "(default: {})").format(
#                               N_IMAGE_COPIES))
# xgroup = parser.add_mutually_exclusive_group()
# xgroup.add_argument("-w", "--windowtime", type=int,
# parser.add_argument("-w", "--windowtime", nargs="?", type=int,
#                     const=DEF_TIME_WINDOW_MINUTES,
#                     default=None,
#                     help=("set time-window [min] " +
#                           "(default: %(default)s, " +
#                           "constant: %(const)s)"))
# ARGS = vars(parser.parse_args())
ARGS = parser.parse_args()
# del(parser)


def initialize_io(dir_in=PATH_IN, dir_out=PATH_OUT,
                  deps=DEPENDENCIES,
                  postfix=POSTFIX, args=ARGS):
    """Set up IO-directories and files and logging."""
    # Determine input head folder
    # if args.interactive or not os.path.isdir(dir_ini):
    if args.interactive or not dir_in.is_dir():
        dir_in = ioh.select_directory(
            title="Input folder containing rpn-snapshot*.csv files",
            # filetypes=[("RPN-log-CSV", "rpn-log_rpn*.csv")],
            dir_ini=dir_in)

        if not dir_in or not dir_in.is_dir():
            print(("No proper input directory: '{}', "
                   "returning 'None'.".format(dir_in)))
            # Leave script
            # sys.exit(1)
            raise SystemExit(1)
            return None

    # Determine output folder
    # Set output level
    # TODO: Remove out_level here?
    out_level = 1  # possibly changed later!
    # if args.interactive or not os.path.isdir(dir_out):
    if args.interactive or not dir_out.is_dir():
        dir_out = ioh.select_directory(
            title="output folder",
            dir_ini=dir_in)

    if not dir_out or not dir_out.is_dir():
        print(("no proper output directory: {}, "
               "creating a sibling " +
               "to input-directory {}").format(
            dir_out, dir_in))
        dir_out = dir_in
        out_level = -1

    # # If output directory is not empty..
    # print(f"Folder '{path_out}' already exists")
    # i = input(f"Clear files in folder '{path_out}' (y/N)?")
    # if i == "y":
    #     # Empty output folder
    #     for the_file in os.listdir(path_out):
    #         file_path = os.path.join(path_out, the_file)
    #         try:
    #             if os.path.isfile(file_path):
    #                 os.unlink(file_path)
    #             # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    #         except Exception as e:
    #             print(e)
    #     print("Files deleted")

    # Set up environment (with holzhelp)
    # hostname = socket.gethostname()
    # NOTE: We want only the name, this also works,
    #       if __file__ returns a full path AND if not
    # thisfile = os.path.basename(__file__)
    thisfile = Path(__file__).name
    dir_out, thisname = ioh.setup_environment(
            thisfile,
            dir_targ=dir_out,
            level=out_level,
            new_dir=True,
            postfix_dir=postfix,
            daystamp=True,
            dependencies=deps,
    )
    ioh.setup_logging(thisname, args, dir_log=dir_out)
    dir_out = Path(dir_out)
    logging.info(f"input from {dir_in}, output to {dir_out}")

    # Display Python version and system info
    # TODO: Move this into holzhelp when doing overhaul
    logging.info(f"Python {platform.python_version()} " +
                 f"on {platform.uname()}")
    # osys = platform.system()
    # if osys.lower() == "windows":
    #     os_ver = platform.win32_ver()[1]
    # else:
    #     # os_ver = platform.linux_distribution()  # NOTE: Deprecated!
    #     import distro
    #     os_ver = distro.linux_distribution()

    # Display versions of used third-party libraries
    # logging.info("matplotlib version: {}".format(matplotlib.__version__))
    # logging.info(f"matplotlib version: {matplotlib.__version__}")
    logging.info(f"numpy version: {np.__version__}")
    logging.info(f"pandas version: {pd.__version__}")
    # logging.info(f"networkx version: {nx.__version__}")
    # logging.info(f"OpenCV version: {cv.__version__}")

    return dir_in, dir_out


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


def convert2utc(dt_naive, local_tz=LOCAL_TZ):
    """Localize naive time and convert it to an aware datetime in UTC.

    LOCAL_TZ = pytz.timezone("Etc/UTC")
    LOCAL_TZ = pytz.timezone("Europe/Vienna")
    """
    # Localize (i.e., make aware)
    dt_local = local_tz.localize(dt_naive)
    # Convert to UTC
    dt_utc = dt_local.astimezone(pytz.utc)

    return dt_utc


def my_date_parser(t_str, time_fmt=TIME_FMT):
    """Parse a timestring into a datetime.datetime object.

    Required for pandas.read_csv() to parse non-standard timestrings.

    Examples (CSV contains a header and a column labelled "time"):

    df = pd.read_csv(csv_path, index_col="time", parse_dates=True,
                     date_parser=my_date_parser)

    df = pd.read_csv(csv_path, converters={"time": my_date_parser})
    """
    return datetime.strptime(t_str, time_fmt)


def my_path_parser(p_str):
    """Parse a pathstring into a pathlib.Path object.

    Required for pandas.read_csv() to parse pathstrings into Paths.

    Example (CSV contains header and column labelled "path"):

    df = pd.read_csv(csv_path, converters={"path": my_date_parser})
    """
    return Path(p_str)


def extract_background(
        filepaths,  # pandas series of pathlib Paths to the photos
        path_out,
        bg_folder=BG_OUTFOLDER,
        print_modulus=PRINT_MODULUS,
        learning_rate=ARGS.learningrate,
        max_runs=ARGS.maxruns,
        history=ARGS.history,
        shadow=ARGS.shadow,
        var_threshold=ARGS.varthreshold,
        sharpen=ARGS.sharpen,
        adjust_gamma=ARGS.adjustgamma,
        args=ARGS,
):
    """Run Gaussian stuff using code from Hannes Oberreither."""
    n_files = len(filepaths)
    in_folder = filepaths[0].parent
    path_out = path_out / bg_folder
    logging.info(f"Received {n_files} in '{in_folder}', "
                 f"exporting to '{path_out}'")

    logging.info(
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

    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    # Somehow I found the value of `gamma=1.2` to be the best in my case
    inv_gamma = 1.0 / 1.2
    ltable = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # Iterate over all files
    for x in range(n_files):

        # We can loop now through our array of images
        img_path = filepaths[x]

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
        logging.debug(f"img_out: {img_output}")
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
        # NOTE: This only works on the raw image and has no effect!!
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
        # # Using string-method .replace:
        # img_bg = str(img_path).replace(in_folder, path_out)
        # Using pathlib Path method .with_parent():
        # TODO: Use the proper filename, when it's in the dataframe
        bg_path = img_path.with_parent(path_out)
        cv.imwrite(str(bg_path), img_bgmodel)

        # Break if max runs is defined and reached
        if max_runs > 0:
            if x == max_runs:
                break

        if x % print_modulus == 0:
            logging.info(f"Current image: {img_path}\n"
                         f"Runs left: {n_files - x}")

    # END

    return bg_path.parent


def main(file_pattern=INFILE_PATTERN, args=ARGS):
    """Extract the background from large amounts of broodnest photos."""
    # Initialize IO-directories and setup logging
    path_in, path_out = initialize_io()

    # Get Paths to all CSV-files
    csv_list = sorted(path_in.glob(file_pattern))
    logging.info(f"Found {len(csv_list)} files "
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
        bg_path = extract_background(df.path, path_out)

    logging.info("Done.")


if __name__ == "__main__":
    main()  # (args)
