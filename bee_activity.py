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

# TIME-RELATED PARAMETERS
# EXPORT_HOURS_UTC = [2, 8, 14, 20]  # must be sorted!

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

# HISTORY = 200  # Number of Photos to look for

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


def compute_difference(img1, img2, euclid=ARGS.euclid):
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
    # gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    absdiff = cv.absdiff(img1, img2)
    # NOTE: Image is still 3 channel BGR

    # TODO: Threshold absdiff to remove JPEG noise?

    # if euclid:
    #     # Euclidean Distance
    #     # diff = np.sqrt(np.sum(np.power(absdiff, 2)))
    #     diff = np.sqrt(np.sum(absdiff.astype(np.float)**2))
    # else:
    #     # Manhattan Distance
    #     diff = np.sum(absdiff)

    # Manhattan Distance
    diff_man = np.sum(absdiff)
    # Euclidean Distance
    diff_euc = np.sqrt(np.sum(absdiff.astype(np.float)**2))

    # TODO: Normalize by time difference?
    #       No, if you wanna do it, do it later!(?)
    # TODO: Return both distances

    return diff_man, diff_euc


def rel_path(path):  # , level=1):
    """Return relative filepath.

    Folder structure e.g.:
        ".../hive1/rpi1/hive1_rpi1_day-190801/
        raw_hive1_rpi1_190801-230001-utc.jpg"
    """
    # plist = list(path.parents)
    # relpath = path.relative_to(path.parents[level])
    relpath = path.relative_to(path.parent.parent)
    return relpath


def make_row(path1, path2, dt1, dt2, diff_man, diff_euc):
    """Make a list containing all row info.

    For Hive and RPi number, just add the two columns later.

    header = [...]
    """
    # # Get interval duration
    # td = (dt2 - dt1)
    # dur = td.total_seconds()
    # dt_center = dt1 + (td / 2)

    # Shorten paths
    # # relpath1 = rel_path(path1)
    # # relpath2 = rel_path(path2)
    relpath1 = path1.relative_to(path1.parent.parent)
    relpath2 = path2.relative_to(path2.parent.parent)

    row = [
        # dt_center, dur,  # Calculate columns later all at once..
        dt1, dt2,
        # path1, path2,
        relpath1, relpath2,
        diff_man,
        diff_euc,
    ]

    return row


def get_difference_df(
        filelist,
        last_img_dict,
        day,
        path_out,
        # export_hours=EXPORT_HOURS_UTC,
        tol_td=TOLERANCE_TIMEDELTA,
        # history=HISTORY,
):
    """Compute differences between all images in the folder.

    Compare the first file to a previous image (last_img), if one
    is there.
    """
    file0name = filelist[0].name
    # NOTE: Assuming all files in folder are from same Hive and RPi
    hive, rpi = parse_filename(file0name)

    # Unpack last_img_dict
    previous = False
    if last_img_dict is not None:

        last_dt = last_img_dict["time"]
        # last_img = last_img_dict["img"]
        # last_path = last_img_dict["path"]

        # Check whether it's close enough to first one here
        dt0 = file_datetime(file0name)
        if dt0 - last_dt < tol_td:

            # Finish unpacking
            last_img = last_img_dict["img"]
            last_path = last_img_dict["path"]
            # Set Boolean True
            previous = True

    # Pick target hour and set up containers
    x = 0
    # # dt_targ = day.replace(hour=export_hours[x])
    # times = []
    # paths = []
    # # target_dfs = []
    # # failures = []
    rows = []
    for img_path in filelist:
        # Parse timestamp into UTC datetime
        dt = file_datetime(img_path.name)

        # Read file with OpenCV
        img = cv.imread(str(img_path))

        if previous:
            # Check if close enough (in time)
            if dt - last_dt < tol_td:
                d_man, d_euc = compute_difference(last_img, img)
                df_row = make_row(
                    last_path, img_path
                    last_dt, dt,
                    d_man, d_euc,
                )
                rows.append(df_row)
            else:
                last_img = img
                last_dt = dt
                last_path = img_path
                # previous = True  # Here true anyway!
        else:
            # Copying not necessary, because 'img' is overwritten!
            # last_img = img.copy()
            last_img = img
            last_dt = dt
            last_path = img_path
            previous = True



        try:

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
    return target_dfs


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
    logging.debug(f"Filename: {file.name}, rpi={rpi}, hive={hive}")

    # NOTE: Temporary hack to fix wrongly named hive2 files
    # TODO: REMOVE, especially when "hive2" really exists!
    if hive == 2:
        hive = 1
        rpi = 2
        logging.warning(f"Changed Hive and RPi numbers: "
                        f"Filename: {file.name}, rpi={rpi}, hive={hive}")

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


def main(
    file_pattern=INFILE_PATTERN,
    # folder_pattern=INFOLDER_PATTERN,
    tol_td=TOLERANCE_TIMEDELTA,
    args=ARGS,
):
    """Compute difference between cosecutive images and output CSVs."""
    # Initialize IO-directories and setup logging
    path_photos, path_out = initialize_io()

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
    if args_euclid:
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
    img = cv.imread(file)
    logging.info(f"Beginning with Hive{hive}, RPi{rpi}, "
                 f"photo '{file}' taken {dt}")

    for i in range(n_files - 1):
        next_file = filelist[i + 1]
        next_hive, next_rpi, next_dt = parse_filename(next_file.name)
        # next_img = cv.imread(file, cv2.IMREAD_GRAYSCALE)
        next_img = cv.imread(next_file)

        # Check whether next file can be compared to the current file
        # if (hive == next_hive) and (rpi == next_rpi) and ...
        if (rpi == next_rpi) and ((dt_next - dt) < tol_td):

            diff = compute_difference(img, next_img)

            # Make row and append
            # row_cols = ["time1", "time2", "activity", "file1", "file2"]
            row = [dt, next_dt, diff, file.name, next_file.name]
            rows.append(row)

            if dt_next.day > dt.day:
                # Export rows as CSV and empty row list
                if len(rows) > 0:
                    logging.info("Day change, exporting CSV")
                    export_csv(rows, row_cols)
                    rows = []

        else:
            # Export rows as CSV and empty row list
            pass


    # # Build the columns for Hive and Pi number
    # hive_col = [hive] * len(rel_paths)
    # rpi_col = [rpi] * len(rel_paths)

    # Process all subfolders containing broodnest photos
    # Reverse order to get the newest folders first
    folders = sorted(path_photos.glob(folder_pattern))
    #                  key=os.path.getmtime)  # , reverse=True)
    n_folders = len(folders)
    logging.info(f"Number of folders: {n_folders}")


    # Remember last image location to use with next folder
    last_img_dict = None
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
        filelist = sorted(folder.glob(file_pattern))
        #                   key=os.path.getmtime)
        logging.info(f"Found {len(filelist)} files.")

        # Go through day-folder and compute differences
        diff_df, last_img_dict = get_difference_df(
                filelist, last_img_dict, day, path_out
        )

        # Export CSV
        export_csv(diff_df, path_out)



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


if __name__ == "__main__":
    main()  # (args)
