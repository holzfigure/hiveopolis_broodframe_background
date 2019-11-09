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
# import cv2 as cv

# Own libraries
import iohelp as ioh

# SETTINGS
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
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
    "broodnest_bgs/csv")
PATH_OUT = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
    "broodnest_bgs"
)
# Filename e.g.:  pi1_hive1broodn_15_8_0_0_4.jpg
INFILE_PATTERN = "pi*_hive*broodn_*.jpg"
# Foldername e.g.:  Photos_of_Pi1_1_9_2019
# Foldername e.g.:  Photos_of_Pi1_heating_1_11_2019
INFOLDER_PATTERN = "Photos_of_Pi*/"
OUTCSV_PREFIX = "bgx"

# TIME-RELATED PARAMETERS
EXPORT_HOURS_UTC = [2, 8, 14, 20]  # must be sorted!
TOLERANCE_TIME_SEC = 60 * 60  # 1 hour
YEAR = 2019
# LOCAL_TZ = pytz.timezone("Etc/UTC")
LOCAL_TZ = pytz.timezone("Europe/Vienna")
TIME_FMT = "%y%m%d-%H%M%S-utc"
TIME_TARGET_FMT = "%y%m%d-%H"
DAY_FMT = "day-%y%m%d"
TIME_INFILE_FMT = "%d_%m_%H_%M_%S.jpg"
TIME_INFOLDER_FMT = "%d_%m_%Y"

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

# argument parsing
parser = argparse.ArgumentParser(
    description=("Extract the broodnest from colony photos."))
parser.add_argument("-d", "--debug", action="store_true",
                    help="debug mode")
# parser.add_argument("-l", "--live", action="store_true",
#                     help="keep looking for new files")
# parser.add_argument("-l", "--latexify", action="store_true",
#                     help="format for IEEE paper")
parser.add_argument("-i", "--interactive", action="store_true",
                    help="popup dialog to select files or folders")
# parser.add_argument("-p", "--plot", action="store_true",
#                     help="make various plots of the data")
# parser.add_argument("-a", "--arrest", action="store_true",
#                     help="show plots and wait until closed")
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


def initialize_io(dir_in=PATH_RAW, dir_out=PATH_OUT,
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


# def adjust_gamma(image, ltable):
#     """Apply gamma correction using the lookup table."""
#     return cv.LUT(image, ltable)
#
#
# def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0,
#                  amount=1.0, threshold=0):
#     """Return a sharpened version of the image, using an unsharp mask.
#
#     https://en.wikipedia.org/wiki/Unsharp_masking
#     https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm"""
#     blurred = cv.GaussianBlur(image, kernel_size, sigma)
#     sharpened = float(amount + 1) * image - float(amount) * blurred
#     sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
#     sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
#     sharpened = sharpened.round().astype(np.uint8)
#     if threshold > 0:
#         low_contrast_mask = np.absolute(image - blurred) < threshold
#         # OpenCV4 function copyTo
#         np.copyto(sharpened, image, where=low_contrast_mask)
#     return sharpened


def folder_datetime(foldername, time_infolder_fmt=TIME_INFOLDER_FMT):
    """Parse UTC datetime from foldername.

    Foldername e.g.:  Photos_of_Pi1_1_9_2019/
                      Photos_of_Pi2_heating_1_11_2019/
    """
    # t_str = folder.name.split("Photos_of_Pi")[-1][2:]  # heating!!
    t_str = "_".join(foldername.split("_")[-3:])
    day_naive = datetime.strptime(t_str, time_infolder_fmt)
    # Localize as UTC
    # day_local = local_tz.localize(day_naive)
    # dt_utc = day_local.astimezone(pytz.utc)
    day = pytz.utc.localize(day_naive)

    return day


def file_datetime(
        filename,  # type <str>  # path.name
        year=YEAR,
        local_tz=LOCAL_TZ,
        # filetag=INFILE_TAG,
        time_infile_fmt=TIME_INFILE_FMT,
):
    """Parse UTC datetime object from filename."""
    # Extract the timestring from the filename
    # Filename e.g.:  pi1_hive1broodn_15_8_0_0_4.jpg
    t_str = filename.split("broodn_")[-1]
    # TODO: Make this more robust for full pathlib Path objects?

    # Parse it into a datetime object
    dt_naive = datetime.strptime(
            t_str, time_infile_fmt).replace(year=year)

    # Localize and convert to UTC
    dt_local = local_tz.localize(dt_naive)
    dt_utc = dt_local.astimezone(pytz.utc)

    return dt_utc


def assemble_timestamps(filelist, year=YEAR):
    """Return a list of timestamps of the same length as the filelist."""

    times = []
    # failures = []
    filelist_out = []
    for file in filelist:
        try:
            # parse timestamp

            times.append(file_datetime(file.name, year=year))
            filelist_out.append(file)
        except Exception as err:
            logging.error("Couldn't parse time of file " +
                          f"'{file}': {err}")
            # failures.append(file)

    # if len(failures) > 0:
    #     logging.warning("Failed files:")
    #     for fail in failures:
    #         logging.warning(f"{fail}")

    # Convert to numpy arrays
    # np.array(timestamps).astype("datetime64")
    times = np.array(times)
    filelist_out = np.array(filelist_out)

    # Sort by time
    time_sort = np.argsort(times)
    times = times[time_sort]
    filelist_out = filelist_out[time_sort]

    logging.info(f"Returning {len(times)} files and times")
    return times, filelist_out


def pack_dataframe(dt_targ, times, paths,
                   history=HISTORY,
                   tol_time=TOLERANCE_TIME_SEC,
                   ):
    """Export a pandas dataframe to extract background."""
    dt = times[-1]
    p = paths[-1]
    logging.info(
        f"Received {len(times)} timestamped files. "
        f"Target: {dt_targ}, current time: {dt}, "
        f"current file: {p.parent.name}/{p.name}"
    )

    # Check whether found time is close enough to target
    delta_t = (dt - dt_targ).total_seconds()
    if abs(delta_t) < tol_time:
        # Truncate lists to history-size
        if (len(times) >= history) and (len(paths) >= history):

            # Keep the last "history" elements
            times = times[-history:]
            paths = paths[-history:]

            # Assemble the data in a pandas dataframe
            # pd.DataFrame({'time': times, 'path': paths})
            # pd.DataFrame(np.array([times, paths]).T,
            #              columns=["time", "path"])
            df = pd.DataFrame(
                index=times,
                data=paths,
                columns=["path"],
            )
            df.index.name = "time"
            df.sort_index(inplace=True)
            logging.info(
                f"Successfully built dataframe of shape {df.shape}"
            )
        else:
            logging.error(
                f"Found only {len(times)} of {history} "
                f"required photos. Skipping target '{dt_targ}'!"
            )
            df = None
    else:
        logging.error(
            f"Found time {dt} is too far from target '{dt_targ}': "
            f"abs({delta_t}) > {tol_time} seconds. Skipping target!"
        )
        df = None

    return df


def export_csv(df, dt_targ,
               path_out=PATH_OUT,
               time_fmt=TIME_FMT,
               time_targ_fmt=TIME_TARGET_FMT,
               prefix=OUTCSV_PREFIX,
               ):
    """Write the timestamped filepath array to CSV."""
    # TODO: Build paths and nice names
    file = df["path"][-1]
    time = df.index[-1]
    # folder = file.parent.name
    # Foldername e.g.:  Photos_of_Pi1_1_9_2019
    # Foldername e.g.:  Photos_of_Pi1_heating_1_11_2019
    # Filename e.g.:  pi1_hive1broodn_15_8_0_0_4.jpg
    trunc = file.name.split("broodn")[0]
    rpi = int(trunc.split("pi")[-1][0])
    hive = int(trunc.split("hive")[-1][0])
    logging.debug(f"filename: {file.name}, rpi={rpi}, hive={hive}")

    targ_str = dt_targ.strftime(time_targ_fmt)
    time_str = time.strftime(time_fmt)

    # Set up output folder
    dir_out = path_out / f"csv/hive{hive}/rpi{rpi}"
    if not dir_out.is_dir():
        dir_out.mkdir(parents=True)
        logging.info(f"Created folder '{dir_out}'")

    # Build filename
    fn = f"{prefix}_hive{hive}_rpi{rpi}_targ{targ_str}_{time_str}.csv"
    ffn = ioh.safename((dir_out / fn), "file")

    # Export CSV
    df.to_csv(
            ffn,
            # index_label="time",
            # date_format=time_fmt,
    )
    logging.info(f"Exported CSV to {ffn}")

    return None


def get_target_dfs(
        filelist,
        day,
        path_out,
        export_hours=EXPORT_HOURS_UTC,
        tol_time=TOLERANCE_TIME_SEC,
        history=HISTORY,
):
    """Extract the relevant chunks of files with their timestamps."""
    target_dfs = []
    # Pick target hour and set up containers
    x = 0
    dt_targ = day.replace(hour=export_hours[x])
    times = []
    paths = []
    # failures = []
    for file in filelist:
        try:
            # Parse timestamp into UTC datetime
            dt = file_datetime(file.name, year=day.year)
            times.append(dt)
            paths.append(file)

            # Check if one of the target hours has been reached
            if dt >= dt_targ:
                # Target hour reached

                # Attempt building target_df
                df = pack_dataframe(dt_targ, times, paths)
                if df is not None:
                    target_dfs.append(df)
                    # Export CSV of the timestamped filepaths
                    export_csv(df, dt_targ, path_out)
                else:
                    logging.error(f"Skipped target '{dt_targ}'")

                # Reset containers and switch to next target hour
                if x < len(export_hours) - 1:
                    x += 1
                    dt_targ = day.replace(hour=export_hours[x])
                    times = []
                    paths = []
                else:
                    # Leave loop when last target hour has been reached
                    break

        except Exception as err:
            # TODO: Put the proper exception here!
            logging.error("Couldn't parse time of file " +
                          f"'{file}': {err}")
            # failures.append(file)

    # Check whether last target was reached, else try with last file

    # # if len(failures) > 0:
    # #     logging.warning("Failed files:")
    # #     for fail in failures:
    # #         logging.warning(f"{fail}")
    #
    # # Convert to numpy arrays
    # # np.array(timestamps).astype("datetime64")
    # times = np.array(times)
    # paths = np.array(paths)
    #
    # # Sort by time
    # time_sort = np.argsort(times)
    # times = times[time_sort]
    # paths = paths[time_sort]

    logging.info(f"Returning {len(times)} files and times")
    return times, paths


def main(
    file_pattern=INFILE_PATTERN,
    folder_pattern=INFOLDER_PATTERN,
    # time_infile_fmt=TIME_INFILE_FMT,
    # time_infolder_fmt=TIME_INFOLDER_FMT,
    # export_hours=EXPORT_HOURS_UTC,
    learning_rate=LEARNING_RATE,
    max_runs=MAX_RUNS,
    print_modulus=PRINT_MODULUS,
    history=HISTORY,
    shadow=SHADOW,
    var_threshold=VAR_THRESHOLD,
    sharpen=SHARPEN,
    adjust_gamma=ADJUST_GAMMA,
):
    """Extract the background from large amounts of broodnest photos."""
    # Initialize IO-directories and setup logging
    path_raw, path_out = initialize_io()

    # Process all subfolders containing broodnest photos
    # Reverse order to get the newest folders first
    folders = sorted(path_raw.glob(folder_pattern),
                     key=os.path.getmtime, reverse=True)
    n_folders = len(folders)
    logging.info(f"Number of folders: {n_folders}")

    i = 0
    for folder in folders:
        i += 1
        logging.info(f"Processing folder {i}/{n_folders}: '{folder.name}'")

        # Get the day as UTC datetime object
        # mtime = folder.stat().st_mtime
        day = folder_datetime(folder.name)

        # Assemble preliminary list of files matching the pattern
        # Filename e.g.:  pi1_hive1broodn_15_8_0_0_4.jpg
        # array = sorted(glob.iglob(path_raw + '/*.jpg'),
        #                key=os.path.getmtime, reverse=True)
        filelist = sorted(folder.glob(file_pattern),
                          key=os.path.getmtime)
        logging.info(f"Found {len(filelist)} files.")

        target_dfs = get_target_dfs(filelist, day, path_out)

        # if len(filelist) > history:
        #     # Get the timestamps and a corresponding filelist
        #     timestamps, filelist = assemble_timestamps(
        #             filelist, year=day.year)
        #
        #     for hour in export_hours:
        #
        #         # Get a target datetime object (i.e. the desired hour)
        #         dt_target = day.replace(hour=hour)
        #
        #         # Compute time-difference to all timestamps
        #         d_seconds = []
        #         # TODO: Do this with sth like "apply_func" or so instead?
        #         for ts in timestamps:
        #             d_seconds.append((ts - dt_target).total_seconds())
        #             # d_seconds.append(abs((ts - dt_target).total_seconds()))
        #         # Take absolute time-diffs
        #         d_seconds = np.absolute(d_seconds)
        #
        #         # Find minimum of absolute deltas
        #         min_idx = np.argmin(d_seconds)
        #         abs_delta = d_seconds[min_idx]
        #
        #         if (abs_delta < hour_tolerance) and (min_idx > history):
        #             closest_time = timestamps[min_idx]
        #             closest_file = filelist[min_idx]
        #         # Make sure it's meaningful (closer than THRESH..)
        #
        #         # Make sure it has a long enough HISTORY..
        #
        # else:  # Skip folder
        #     logging.info(
        #         f"WARNING: Folder '{folder}' doesn't contain enough data: "
        #         f"{len(filelist)} files < {history} minimally required."
        #     )



    # logging.info(
    #     "======= SETTINGS =======\n"
    #     f"Path raw:      {path_raw}\n"
    #     f"Path output:   {path_out}\n"
    #     f"Print modulus: {print_modulus}\n"
    #     f"Max runs:      {max_runs}\n"
    #     f"Learning rate: {learning_rate}\n"
    #     f"History:       {history}\n"
    #     f"Shadow:        {shadow}\n"
    #     f"var_threshold: {var_threshold}\n"
    #     f"Sharpen:       {sharpen}\n"
    #     f"Adjust Gamma:  {adjust_gamma}\n"
    # )
    #
    # # Load CV BackgroundSubtractorMOG2 with settings
    # # https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    # # https://docs.opencv.org/master/d6/d17/group__cudabgsegm.html
    # # Attributes found inside Class
    # # 'setBackgroundRatio', 'setComplexityReductionThreshold',
    # # 'setDetectShadows', 'setHistory', 'setNMixtures',
    # # 'setShadowThreshold', 'setShadowValue', 'setVarInit', 'setVarMax',
    # # 'setVarMin', 'setVarThreshold', 'setVarThresholdGen'
    # mog = cv.createBackgroundSubtractorMOG2()
    # mog.setDetectShadows(shadow)
    # mog.setHistory(history)
    # mog.setVarThreshold(var_threshold)
    #
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
    #
    #
    #
    # print("#############################")
    # logging.info("Starting with total file number: " + str(len(array)))
    # print("#############################")
    #
    # # Build a lookup table mapping the pixel values [0, 255] to
    # # their adjusted gamma values
    # # Somehow I found the value of `gamma=1.2` to be the best in my case
    # inv_gamma = 1.0 / 1.2
    # ltable = np.array([
    #     ((i / 255.0) ** inv_gamma) * 255
    #     for i in np.arange(0, 256)]).astype("uint8")
    #
    # # loop x times as files in our folder
    # for x in range(len(array)):
    #
    #     # We can loop now through our array of images
    #     img_path = array[x]
    #
    #     # Read file with OpenCV
    #     img = cv.imread(img_path)
    #
    #     # Preprocessing ######
    #     # Change image to grayscale
    #     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     # Equalize grayscale image
    #     # img = cv.equalizeHist(img)
    #     # Add gaussian to remove noise
    #     # img = cv.GaussianBlur(img, (1, 1), 0)
    #     # img = cv.medianBlur(img, 1)
    #     # img = cv.GaussianBlur(img, (7, 7), 1.5)
    #     # END Preprocessing ######
    #
    #     # Apply algorithm to generate background model
    #     img_output = mog.apply(img, learning_rate)
    #     logging.info(f"img_out: {img_output}")
    #     # Threshold for foreground mask, we don't use the
    #     # foreground mask so we dont need it?
    #     # img_output = cv.threshold(img_output, 10, 255, cv.THRESH_BINARY);
    #
    #     # Get background image
    #     img_bgmodel = mog.getBackgroundImage()
    #
    #     # Postprocessing ######
    #
    #     # Sharpen the image
    #     if sharpen:
    #         img_bgmodel = unsharp_mask(img_bgmodel)
    #
    #     # Adjust gamma if there is light change
    #     if adjust_gamma:
    #         img = adjust_gamma(img, ltable)
    #
    #     # Change image to grayscale
    #     # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     # Equalize grayscale image
    #     # img = cv.equalizeHist(img)
    #     # Add gaussian to remove noise
    #     # img = cv.GaussianBlur(img, (1, 1), 0)
    #     # img = cv.medianBlur(img, 1)
    #     # img = cv.GaussianBlur(img, (7, 7), 1.5)
    #
    #     # img_bgmodel = cv.equalizeHist(img_bgmodel)
    #     # END Preprocessing ######
    #
    #     # Write finished backgroundModels
    #     img_bg = img_path.replace(path_raw, path_out)
    #     cv.imwrite(img_bg, img_bgmodel)
    #
    #     # Break if max runs is defined and reached
    #     if max_runs > 0:
    #         if x == max_runs:
    #             break
    #
    #     if (x % logging.info_modulus) == 0:
    #         logging.info(f"Current image: {img_path}\nRuns left: {len(array)-x}")
    #
    # # END


if __name__ == "__main__":
    main()  # (args)
