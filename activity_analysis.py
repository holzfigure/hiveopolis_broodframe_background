#!/usr/bin/env python3
"""Analyse and plot the computed image differences.

Iterate over CSVs containing differences between consecutive
broodnest photos. Parse the contents into pandas DataFrames
and make different plots, e.g. boxplots and functional boxplots,
the raw data as lines in different colors (e.g. along a colorscale,
with earlier ones being brighter (summer,), regression? sine function?
show the data as one long sine function instead of superimposing
the days).. Do a Fourier transform of it to get the period and see it's 24hs

Compare Euclidean to Manhattan distance?

 and compute the distance between
consecutive images, output the information as a CSV file
containing in each row the times and filenames for both images,
the hive and RPi number and the difference score.

Euclidean distance or absolute difference are implemented.
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
# import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt

# Own libraries
import iohelp as ioh

# IO-SETTINGS
POSTFIX = None
DEPENDENCIES = [
    Path("iohelp.py"),
    # Path("iohelp/iohelp_tk.py"),
    # Path("iohelp/ioplot.py"),
]
# Path to raw files and output folder of generated images
# PATH_RAW = Path.cwd() / 'img'
# PATH_OUT = Path.cwd() / 'out'
# PATH_PHOTOS = Path(
#     "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/"
#     "broodnest_obs/hive1"
# )
PATH_CSVS = Path(
    "/media/holzfigure/Data/local_stuff/Hiveopolis/broodnests/"
    "bee_activity_csvs/csv/"  # hive1/rpi1/"
)
# PATH_OUT = Path(
#     "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
#     "broodnest_activity/csv"
# )
PATH_OUT = Path(
    "/media/holzfigure/Data/local_stuff/Hiveopolis/broodnests/"
    "bee_activity"
)
# Filename e.g.: "act_hive1_rpi1_190804_000000-235959-utc_euclidean.csv"
INFILE_PATTERN = "act_hive*_rpi*-utc*.jpg"
# Foldername e.g.:  Photos_of_Pi1_1_9_2019
# Foldername e.g.:  Photos_of_Pi1_heating_1_11_2019
# INFOLDER_PATTERN = "hive*_rpi*_day-*/"
OUT_PREFIX = "act"

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
TIME_INFILE_FMT = "%y%m%d_%H%M%S-%H%M%S-utc"  # 2nd part is time-span

# argument parsing
parser = argparse.ArgumentParser(
    description=("Extract the broodnest from colony photos."))
parser.add_argument("-d", "--debug", action="store_true",
                    help="debug mode")
parser.add_argument("-i", "--interactive", action="store_true",
                    help="popup dialog to select files or folders")
# parser.add_argument("-e", "--euclid", action="store_true",
#                     help=("compute Euclidean distance between images "
#                           "(default: Manhattan distance)"))
# parser.add_argument("-f", "--firstidx", type=int,
#                     default=0,
#                     help=("file index from where to start " +
#                           "(default:  %(default)s)"))
# parser.add_argument("--export", action="store_true",
#                     help="export the difference images")
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


def initialize_io(dir_in=PATH_CSVS, dir_out=PATH_OUT,
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
    logging.info(f"matplotlib version: {matplotlib.__version__}")
    logging.info(f"numpy version: {np.__version__}")
    logging.info(f"pandas version: {pd.__version__}")
    # logging.info(f"networkx version: {nx.__version__}")
    # logging.info(f"OpenCV version: {cv.__version__}")

    return dir_in, dir_out


def parse_filename(filename):  # , time_fmt=TIME_INFILE_FMT):
    """Parse Hive and RPi number from filename.

    Filename e.g.:
    "act_hive1_rpi1_190804_000000-235959-utc_euclidean.csv"
    """
    # Split the name up into its "blocks"
    parts = filename.split("_")
    hive_str, rpi_str = parts[1:3]
    method = parts[5]

    # Parse Hive and RPi number
    hive = int(hive_str[-1])
    rpi = int(rpi_str[-1])
    method = method.strip(".csv")

    # # Parse timestring into a datetime object
    # dt_naive = datetime.strptime(t_str, time_fmt)
    # dt_utc = pytz.utc.localize(dt_naive)

    return hive, rpi, method


def main(
    file_pattern=INFILE_PATTERN,
    # folder_pattern=INFOLDER_PATTERN,
    tol_td=TOLERANCE_TIMEDELTA,
    args=ARGS,
):
    """Read image-difference CSVs into dataframes and make plots."""
    # Initialize IO-directories and setup logging
    path_in, path_out = initialize_io()

    # path_diffs = path_out / "diff_imgs"
    # if args.export:
    #     # Folder not needed otherwise, but variable needs to be passed
    #     if not path_diffs.is_dir():
    #         path_diffs.mkdir()
    #         logging.info(f"Created folder '{path_diffs}'")

    # Find matching files
    # NOTE: This can take potentially long
    #       A folderwise sorting would be much faster
    # t0 = time.time()
    filelist = sorted(path_in.rglob(file_pattern))
    # dur = time.time() - t0

    n_files = len(filelist)
    logging.info(f"Found {n_files} matching files in '{path_in}'")
    #              f"(took {dur:.4} seconds)")

    df_list = []
    for csv_path in filelist:
        logging.info(f"Reading '{csv_path.name}'")

        hive, rpi, method = parse_filename(csv_path.name)
        # Read CSV
        # See https://pandas.pydata.org/pandas-docs/stable/reference/
        # api/pandas.read_csv.html
        # df = pd.read_csv(csv_path, index_col="time", parse_dates=True,
        #                  date_parser=my_date_parser)
        # Works only with the default pandas time format:
        df = pd.read_csv(
                csv_path,
                index_col="time_central",
                parse_dates=["time_central", "time1", "time2"],
                # converters={"path": my_path_parser}),
        )

        df_list.append(df)

    try:
        pass

    except KeyboardInterrupt:
        logging.info("Manually interrupted script")

    finally:
        # if len(rows) > 0:
        #     logging.info(f"Exporting {len(rows)} rows to CSV")
        #     export_csv(rows, row_cols, path_out, hive, rpi, method)

        logging.info("Done.")


if __name__ == "__main__":
    main()
