#!/usr/bin/env python3
"""Find relevant broodnest photos for background extraction.

Iterate over all photos and compute the distance between
consecutive images, output the information as a CSV file
containing in each row the times and filenames for both images,
the hive and RPi number and the difference score.

Euclidean distance or absolute difference will be implemented.
"""

# Basic libraries
# import os
# import glob
import time
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime, timedelta

# Third-party libraries
import pytz
import pandas as pd
import numpy as np
# OpenCV, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
# NOTE: OpenCV 3.2 works on Ubuntu 18.04
import cv2 as cv

# Own libraries
import iohelp as ioh

# SETTINGS
POSTFIX = None
DEPENDENCIES = [
    Path("iohelp.py"),
    # Path("iohelp/iohelp_tk.py"),
    # Path("iohelp/ioplot.py"),
]
# Path to raw files and output folder of generated images
# PATH_RAW = Path.cwd() / 'img'
# PATH_OUT = Path.cwd() / 'out'
PATH_PHOTOS = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/"
    "broodnest_obs/hive1")
PATH_OUT = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
    "broodnest_activity/csv"
)
# Filename e.g.:  pi1_hive1broodn_15_8_0_0_4.jpg
INFILE_PATTERN = "raw_hive*_rpi*-utc.jpg"
# Foldername e.g.:  Photos_of_Pi1_1_9_2019
# Foldername e.g.:  Photos_of_Pi1_heating_1_11_2019
INFOLDER_PATTERN = "hive*_rpi*_day-*/"
OUTCSV_PREFIX = "act"
OUTRAW_PREFIX = "raw"

# Maximal seconds accepted between images:
TOLERANCE_TIMEDELTA = timedelta(seconds=20)

LOCAL_TZ = pytz.timezone("Etc/UTC")
# LOCAL_TZ = pytz.timezone("Europe/Vienna")
TIME_FMT = "%y%m%d-%H%M%S-utc"
START_TIME_FMT = "%y%m%d-%H%M%S"
END_TIME_FMT = "%H%M%S-utc"
# TIME_TARGET_FMT = "%y%m%d-%H"
# DAY_FMT = "day-%y%m%d"
# TIME_INFILE_FMT = "%d_%m_%H_%M_%S.jpg"
TIME_INFOLDER_TAG = "day-"
TIME_INFOLDER_FMT = "%y%m%d"
TIME_INFILE_TAG = "-utc"
TIME_INFILE_FMT = "%y%m%d-%H%M%S-utc.jpg"

# argument parsing
parser = argparse.ArgumentParser(
    description=("Extract the broodnest from colony photos."))
parser.add_argument("-d", "--debug", action="store_true",
                    help="debug mode")
parser.add_argument("-i", "--interactive", action="store_true",
                    help="popup dialog to select files or folders")
parser.add_argument("-e", "--euclid", action="store_true",
                    help=("compute Euclidean distance between images "
                          "(default: Manhattan distance)"))
parser.add_argument("-f", "--firstidx", type=int,
                    default=0,
                    help=("file index from where to start " +
                          "(default:  %(default)s)"))
parser.add_argument("--export", action="store_true",
                    help="export the difference images")
# xgroup = parser.add_mutually_exclusive_group()
# xgroup.add_argument("-", "--windowtime", type=int,
# parser.add_argument("-w", "--windowtime", nargs="?", type=int,
#                     const=DEF_TIME_WINDOW_MINUTES,
#                     default=None,
#                     help=("set time-window [min] " +
#                           "(default: %(default)s, " +
#                           "constant: %(const)s)"))
# ARGS = vars(parser.parse_args())
ARGS = parser.parse_args()
# del(parser)


def initialize_io(dir_in=PATH_PHOTOS, dir_out=PATH_OUT,
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
    logging.info(f"pytz version: {pytz.__version__}")
    # logging.info("matplotlib version: {}".format(matplotlib.__version__))
    # logging.info(f"matplotlib version: {matplotlib.__version__}")
    logging.info(f"numpy version: {np.__version__}")
    logging.info(f"pandas version: {pd.__version__}")
    # logging.info(f"networkx version: {nx.__version__}")
    logging.info(f"OpenCV version: {cv.__version__}")

    return dir_in, dir_out


def parse_filename(filename, time_fmt=TIME_INFILE_FMT):
    """Parse Hive and RPi number from filename.

    Filename e.g.:  raw_hive1_rpi1_190801-000002-utc.jpg
    """
    # Split the name up into its "blocks"
    prefix, hive_str, rpi_str, t_str = filename.split("_")

    # Parse Hive and RPi number
    hive = int(hive_str[-1])
    rpi = int(rpi_str[-1])

    # Parse timestring into a datetime object
    dt_naive = datetime.strptime(t_str, time_infile_fmt)
    # dt_utc = pytz.utc.localize(dt_naive)

    return hive, rpi, dt_naive


def pack_dataframe(dt_targ, times, paths,
                   history=HISTORY,
                   tol_time=TOLERANCE_TIME_SEC,  # TOLERANCE_TIMEDELTA
                   ):
    """Export a pandas dataframe to extract background.
    """
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

            paths, names, hives, rpis = convert_paths(paths, times)

            # Assemble the data in a pandas dataframe
            # pd.DataFrame({'time': times, 'path': paths})
            # pd.DataFrame(np.array([times, paths]).T,
            #              columns=["time", "path"])
            df = pd.DataFrame(
                index=times,
                data=np.array([hives, rpis, paths, names]).T,
                columns=["hive", "rpi", "path", "name"],
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


def compute_difference(img1, img2, path_out,
                       euclid=ARGS.euclid,
                       export=ARGS.export,
                       time_fmt=TIME_FMT,
                       ):
    """Compute difference between two images.

    def euclid(img1,img2):
        absdiff = cv.absdiff(img1, img2).astype(np.float)
        return np.sqrt(np.sum(absdiff**2))

    def manhattan(img1,img2):
        return np.sum(cv.absdiff(img1,img2))

    def time_it(img1, img2, n=10):
        t0=time.time()
        for i in range(n):
            diff = euclid(img1, img2)
        t1=time.time()
        for i in range(n):
            diff = manhattan(img1, img2)
        t2=time.time()
        print(f"{n} euclid took {t1 - t0} secs, manhatten {t2 - t1}")

    >>> time_it(img1, img2, 1000)
    1000 euclid took 19.7092 secs, manhatten 2.6678

    >>> 19.7092 / 2.6679
    7.3875

    euclid takes about 7.5 times as long..

    Useful links:

    https://stackoverflow.com/questions/56183201/
    detect-and-visualize-differences-between-two-images-with-opencv-python/
    56193442

    https://stackoverflow.com/questions/27035672/
    cv-extract-differences-between-two-images/27036614#27036614
    """
    # Get absolute differences for each pixel as UINT8 array
    # TODO: Turn grayscale first?
    #       No! Then read them in as grayscale in the first place!
    # gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    absdiff = cv.absdiff(img1, img2)
    # NOTE: Image is still 3 channel BGR

    if export:
        t_str = datetime.utcnow().strftime(time_fmt)
        ffn = path_out / f"diff_{t_str}.png"
        cv.imwrite(ffn)

    # TODO: Threshold absdiff to remove JPEG noise?

    if euclid:
        # Euclidean Distance
        # diff = np.sqrt(np.sum(np.power(absdiff, 2)))
        diff = np.sqrt(np.sum(absdiff.astype(np.float)**2))
    else:
        # Manhattan Distance
        diff = np.sum(absdiff)
    #
    # # Manhattan Distance
    # diff_man = np.sum(absdiff)
    # # Euclidean Distance
    # diff_euc = np.sqrt(np.sum(absdiff.astype(np.float)**2))

    # TODO: Normalize by time difference?
    #       No, if you wanna do it, do it later!(?)
    # TODO: Return both distances

    return diff


    # # Sort by time
    # time_sort = np.argsort(times)
    # times = times[time_sort]
    # paths = paths[time_sort]


def export_csv(rows, row_cols, path_out,
               time_fmt=TIME_FMT,
               prefix=OUTCSV_PREFIX,
               ):
    """Write the image difference to CSV."""

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


def main(
    file_pattern=INFILE_PATTERN,
    # folder_pattern=INFOLDER_PATTERN,
    tol_td=TOLERANCE_TIMEDELTA,
    args=ARGS,
):
    """Compute difference between cosecutive images and output CSVs."""
    # Initialize IO-directories and setup logging
    path_photos, path_out = initialize_io()

    path_diffs = path_out / "diff_imgs"
    if args.export:
        # Folder not needed otherwise, but variable needs to be passed
        if not path_diffs.is_dir():
            path_diffs.mkdir()
            logging.info(f"Created folder '{path_diffs}'")

    # Find matching files
    # NOTE: This can take potentially long
    #       A folderwise sorting would be much faster
    t0 = time.time()
    filelist = sorted(path_photos.rglob(file_pattern))
    dur = time.time() - t0

    n_files = len(filelist)
    logging.info(f"Found {n_files} matching files in '{path_photos}' "
                 f"(took {dur:.4} seconds)")

    # Trim list according to given first_idx
    if args.firstidx is not None:
        filelist = filelist[args.firstidx:]
        n_files = len(filelist)
        logging.info(f"Trimmed filelist to {n_files} files")

    # Log differencing method employed
    if args.euclid:
        method = "Euclidean"
    else:
        method = "Manhattan"
    logging.info(f"Computing {method} distance between images")

    # Initialize containers
    row_cols = [
        "time1", "time2", "activity", "file1", "file2"
    ]
    rows = []

    # Begin clocking
    t0 = time.time()

    # Parse first file
    file = filelist[0]
    # c_dir1 = c_file.parent
    hive, rpi, dt = parse_filename(file.name)
    # img = cv.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv.imread(str(file))
    logging.info(f"Beginning with Hive{hive}, RPi{rpi}, "
                 f"photo '{file}' taken {dt}")

    try:
        for i in range(n_files - 1):
            next_file = filelist[i + 1]
            next_hive, next_rpi, next_dt = parse_filename(next_file.name)
            # next_img = cv.imread(file, cv2.IMREAD_GRAYSCALE)
            next_img = cv.imread(str(next_file))

            # Check whether next file can be compared to the current file
            # if (hive == next_hive) and (rpi == next_rpi) and ...
            if (rpi == next_rpi) and ((next_dt - dt) < tol_td):

                diff = compute_difference(img, next_img, path_diffs)

                # Make row and append
                # row_cols = ["time1", "time2", "activity", "file1", "file2"]
                row = [dt, next_dt, diff, file.name, next_file.name]
                rows.append(row)

                if next_dt.day > dt.day:
                    # Export rows as CSV and empty row list
                    if len(rows) > 0:
                        logging.info("Day change, "
                                     f"exporting {len(rows)} to CSV")
                        export_csv(rows, row_cols, path_out)
                        rows = []

            else:
                logging.info(
                    "Photos not comparable: "
                    f"file1: {file.name}, file2: {next_file.name}, "
                    "switching to next series"
                )
                # Export rows as CSV and empty row list
                if len(rows) > 0:
                    logging.info(f"Exporting {len(rows)} rows to CSV")
                    export_csv(rows, row_cols, path_out)
                    rows = []
                pass

            # Reset current photo data
            file, dt, img = next_file, next_dt, next_img
            hive, rpi = next_hive, next_rpi

    except KeyboardInterrupt:
        logging.info("Manually interrupted script")

    finally:
        if len(rows) > 0:
            logging.info(f"Exporting {len(rows)} rows to CSV")
            export_csv(rows, row_cols, path_out)

    # # Build the columns for Hive and Pi number
    # hive_col = [hive] * len(rel_paths)
    # rpi_col = [rpi] * len(rel_paths)


if __name__ == "__main__":
    main()  # (args)
