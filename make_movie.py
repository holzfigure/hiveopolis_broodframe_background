#!/usr/bin/env python3
"""
Make a video from images.

https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/
reading_and_writing_images.html#imread

Valid file formats (OpenCV 3):
    Windows bitmaps - *.bmp, *.dib (always supported)
    JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section)
    JPEG 2000 files - *.jp2 (see the Notes section)
    Portable Network Graphics - *.png (see the Notes section)
    WebP - *.webp (see the Notes section)
    Portable image format - *.pbm, *.pgm, *.ppm (always supported)
    Sun rasters - *.sr, *.ras (always supported)
    TIFF files - *.tiff, *.tif (see the Notes section)

valid_image_formats = [
    ".bmp", ".dib",
    ".jpeg", ".jpg", ".jpe", ".jp2",
    ".png", ".webp",
    ".pbm", ".pgm", ".ppm",
    ".sr", ".ras",
    ".tiff", ".tif"
]

Notes:
    The function determines the type of an image by the content,
    not by the file extension.

    On Microsoft Windows* OS and MacOSX*, the codecs shipped with
    an OpenCV image (libjpeg, libpng, libtiff, and libjasper) are
    used by default. So, OpenCV can always read JPEGs, PNGs, and TIFFs.
    On MacOSX, there is also an option to use native MacOSX image readers.
    But beware that currently these native image loaders give images
    with different pixel values because of the color management
    embedded into MacOSX.

    On Linux*, BSD flavors and other Unix-like open-source
    operating systems, OpenCV looks for codecs supplied with an OS image.
    Install the relevant packages (do not forget the development files,
    for example, “libjpeg-dev”, in Debian* and Ubuntu*) to get the
    codec support or turn on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake.
"""
import re
import time
# import shutil
import logging
import platform
import argparse
from statistics import median
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np

import iohelp as ioh

import warnings
warnings.filterwarnings('error', category=DeprecationWarning)

# # GLOBALS
# TODO: get dependency filenames from the imported modules
DEPENDENCIES = [
        Path("iohelp.py").resolve(),
        # Path("holzhelp/holzplot.py"),
        ]

DIR_IN = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/"
    "broodnest_bgs"
)
DIR_OUT = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/"
    "broodnest_bgs"
)

POSTFIX_DIR = "vid"
VID_PREFIX = "v_"
# LINK_PREFIX = "im"
# TIMEZONE = "Europe/Vienna"
TIME_FMT = "%y%m%d-%H%M%S-utc"
TIMESTAMP_FMT = "%y-%m-%d %H:%M UTC"
DAY_FMT = "%y%m%d"
TIME_MATCH = "-utc"
TIME_FMT_TRIMMED = TIME_FMT.split(TIME_MATCH)[0]
FN_TERM_SEPARATOR = "_"

DEF_IMG_PATTERN = "*.jpg"
DEF_IMG_EXT = ".jpg"
DEF_VID_EXT = ".avi"
STRIP_CHARS = " -_()"

MINIMAL_FILESIZE = 1000  # bytes  # 1500000
DEF_FRAMERATE = 30
DEF_REDUCE = 1
DEF_MULTIPLY = 1
DEF_INTERVAL = 1

ADD_LAST_IMAGE_SECONDS = 3
SKIP_FRAMES_INFO = 50
# WAIT_BEFORE_LINK_DELETION = 10

QUALITY = 23  # OpenH264

# argument parsing
# TODO: set -n -f -d to optionally accept an input parameter
parser = argparse.ArgumentParser(
        description="Make a video from image files.")
parser.add_argument("-d", "--debug", action="store_true",
                    help="debug mode")
parser.add_argument("-i", "--interactive", action="store_true",
                    help="popup dialog to select file")
# parser.add_argument("-p", "--png", action="store_true",
#                     help="use png instead of default jpg images")
parser.add_argument("-c", "--crazy", action="store_true",
                    help=("allow searching for images with " +
                          "unorthodox filenames"))
parser.add_argument("-a", "--all", action="store_true",
                    help=("plot all images. if not set, you will " +
                          "be prompted for start and end times"))
parser.add_argument("-n", "--name", action="store_true",
                    help="change default video-filename")
parser.add_argument("-t", "--timestamp", action="store_true",
                    help="timestamp video")
parser.add_argument("-p", "--pattern", type=str,
                    default=DEF_IMG_PATTERN,
                    help=(f"define search pattern [e.g.: ''*.jpg'], " +
                          f"extension will be parsed from it " +
                          f"(default: %(default)s)"))
# parser.add_argument("-x", "--extension", type=str,
#                     default=DEF_EXT,
#                     help=("image format " +
#                           f"(default: {DEF_EXT})"))
# parser.add_argument("-f", "--framerate", action="store_true",
#                     help="change default framerate")
parser.add_argument("-f", "--framerate", type=int,
                    default=DEF_FRAMERATE,
                    help=("change default framerate " +
                          f"(default: {DEF_FRAMERATE})"))
parser.add_argument("-v", "--interval", type=int,
                    default=DEF_INTERVAL,
                    help=("define interval in minutes " +
                          "between consecutive shots " +
                          f"(default: {DEF_INTERVAL})"))
x_group = parser.add_mutually_exclusive_group(required=False)
x_group.add_argument("-r", "--reduce", type=int,
                     default=DEF_REDUCE,
                     help=("reduce every REDUCE frame " +
                           f"(default: {DEF_REDUCE})"))
x_group.add_argument("-m", "--multiply", type=int,
                     default=DEF_MULTIPLY,
                     help=("prompt for frame multiplication " +
                           f"(default: {DEF_MULTIPLY})"))
ARGS = parser.parse_args()


def initialize_io(
        dir_ini=DIR_IN, dir_out=DIR_OUT,
        deps=DEPENDENCIES,
        postfix=POSTFIX_DIR, args=ARGS):
    """Set up IO-directories and files and logging."""
    # Determine input folder
    # if args.interactive or not os.path.isdir(dir_ini):
    if args.interactive or not dir_ini.is_dir():
        dir_in = ioh.select_directory(
            title="input folder containing images",
            dir_ini=dir_ini)
        # # NOTE: ioh.select_dictionary return a pathlib.Path
        # dir_in = Path(dir_in)

        if not dir_in or not dir_in.is_dir():
            print(("No proper input directory: '{}', "
                   "aborting...".format(dir_in)))
            raise SystemExit(1)
            # return None
    else:
        dir_in = dir_ini

    # Determine output folder
    if args.interactive:
        dir_out = ioh.select_directory(
            title="output folder",
            dir_ini=dir_in)
        # Make sure directory is valid
        if not dir_out or not dir_out.is_dir():
            print(("No proper output directory: '{}', "
                   "aborting...".format(dir_in)))
            raise SystemExit(1)
        out_level = 1  # As subdirectory in the given directory
    else:
        dir_out = dir_in
        out_level = -1  # As sibling to the given directory

    # setup environment (with iohelp)
    # hostname = socket.gethostname()
    # thisfile = os.path.basename(__file__)
    # NOTE: We want only the name, this also works,
    #       if __file__ returns a full path AND if not
    thisfile = Path(__file__).name
    dir_out, thisname = ioh.setup_environment(thisfile, dir_targ=dir_out,
                                              level=out_level, new_dir=True,
                                              postfix_dir=postfix,
                                              daystamp=True,
                                              dependencies=deps)
    ioh.setup_logging(thisname, args, dir_log=dir_out)
    dir_out = Path(dir_out)
    logging.info(f"input from {dir_in}, output to {dir_out}")

    # Display Python version and system info
    # TODO: Move this into holzhelp when doing overhaul
    logging.info(f"Python {platform.python_version()} " +
                 f"on {platform.uname()}")

    # Display versions of used third-party libraries
    # logging.info("matplotlib version: {}".format(matplotlib.__version__))
    # logging.info(f"matplotlib version: {matplotlib.__version__}")
    # logging.info(f"pandas version: {pd.__version__}")
    logging.info(f"numpy version: {np.__version__}")
    logging.info(f"OpenCV version: {cv2.__version__}")

    return dir_in, dir_out


def correct_img_pattern(pattern,
                        def_ext=DEF_IMG_EXT,
                        def_pattern=DEF_IMG_PATTERN, args=ARGS):
    """Check the search pattern and possibly correct it.

    TODO: Do something about crazy mode?
    """
    valid_image_formats = [
        ".bmp", ".dib",
        ".jpeg", ".jpg", ".jpe", ".jp2",
        ".png", ".webp",
        ".pbm", ".pgm", ".ppm",
        ".sr", ".ras",
        ".tiff", ".tif"
    ]

    # Check if passed pattern is not empty
    if len(pattern) > 0:

        # Define name and extension
        # if "." in pattern:
        parts = pattern.split(".")
        n_parts = len(parts)
        if n_parts == 2:
            # Single dot (1 name, 1 extension)
            name = parts[0]
            ext = "." + parts[1]
        elif n_parts > 2:
            # Multiple dots in pattern, join first parts
            name = ".".join(parts[:-1])
            ext = "." + parts[-1]
        elif n_parts == 1:
            # No dot in pattern
            name = parts[0]
            if args.crazy:
                # Allow looking for all filenames
                ext = ""
            else:
                logging.warning(f"No extension in pattern '{pattern}', " +
                                f"adding default '{def_ext}'.")
                ext = def_ext

        # Check if extension is valid for OpenCV 3 cv2.imread()
        if (ext not in valid_image_formats) and (not args.crazy):
            logging.warning(f"Invalid search pattern extension '{ext}', "
                            f"replacing with default '{def_ext}'.")
            ext = def_ext

        # Reassemble pattern
        pattern = name + ext
    else:
        # Passed empty pattern
        logging.warning(f"Empty search pattern passed," +
                        f"replacing with default '{def_pattern}'.")
        pattern = def_pattern

    return pattern, ext


def time_from_filename_cautious(filepath,
                                time_fmt=TIME_FMT_TRIMMED,
                                match=TIME_MATCH,
                                sep=FN_TERM_SEPARATOR,
                                strip_chars=STRIP_CHARS,
                                args=ARGS):
    """Parse the UTC time from a standardized (full) filename.

    Format should be like "XXX_%y%m%d-%H%M%S_utcXXXX".
    Full paths to such files also work.

    Outputs a naive datetime object in UTC time.
    """
    # dtimestr = fn.split("/")[-1].split("-utc")[0].split("_")[-1]
    p = Path(filepath).resolve()

    stem = p.stem
    parts = stem.split(match)
    # n_parts = len(parts)
    datetime_obj = None
    if len(parts) > 1:
        # Match found
        # parts = parts[0].split(sep)
        # if len(parts) > 1:
        #     # Chopped off parts before time_str
        #     time_str = parts[-1]
        # else:
        #     # No earlier part found
        #     time_str = parts[0]
        parts = parts[0].split(sep)
        time_str = parts[-1]
        name = parts[0]
        try:
            datetime_obj = datetime.strptime(time_str, time_fmt)
        except ValueError as err:
            logging.warning(f"Could not parse time_str {time_str}! " +
                            f"Error: {err}")
    else:
        # No match found
        datetime_obj = None

        # Look for trailing number
        # https://docs.python.org/3.7/library/re.html
        # https://docs.python.org/3/howto/regex.html
        stem = stem.rstrip(strip_chars)
        m = re.search(r'\d+$', stem)
        if m:
            # TODO: Store the number? (not here but
            #       in a function for the loop)
            # Strip number from name
            raw_name = stem.split(m.group())[0]
            # Trim off dashes or whitespace
            name = raw_name.rstrip(strip_chars)
        else:
            name = stem

    return datetime_obj, name


def time_from_filename_fast(filepath,
                            time_fmt=TIME_FMT_TRIMMED,
                            match=TIME_MATCH,
                            sep=FN_TERM_SEPARATOR):
    """Parse the UTC time from a standardized (full) filename.

    Expects a pathlib.Path as 'fn'
    Format should be like "XXX_%y%m%d-%H%M%S-utcXXX".
    Full paths to such files also work.

    Outputs a naive datetime object in UTC time.
    """
    # dtimestr = fn.split("/")[-1].split("_utc")[0].split("_")[-1]
    # fn = Path(fn).resolve()
    time_str = filepath.stem.split(match)[0].split(sep)[-1]
    return datetime.strptime(time_str, time_fmt)


def date_duration_string(dt0, dtx, time_fmt=DAY_FMT):
    """Generate duration string for filename."""
    str0 = dt0.strftime(time_fmt)
    strx = dtx.strftime(time_fmt)
    # day_idx = len(time_fmt.split("-")[0])
    # if str0[:day_idx] == strx[:day_idx]:
    #     # Day is the same, use whole string?
    #     # akh..
    #     # fwk it..
    #     pass
    # dur_str = f"{str0[:day_idx]}-{strx[:day_idx]}"
    dur_str = f"{str0}-{strx}"
    return dur_str


def get_time_info(paths,
                  match=TIME_MATCH, full_fmt=TIME_FMT,
                  args=ARGS):
    """Attempt to parse filenames for timestrings.

    Look for a match (probably sth like 'utc' or '-utc'),
    split everything off behind it and then split off everything
    before the last underscore '_'.

    https://docs.python.org/3.6/library/
    datetime.html#datetime.datetime.timestamp

    Python > 3.3:
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()

    or:
    timestamp = (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)

    """
    # trim_list = ["-", "_"]
    # fmt = full_fmt.split(match)[0]
    # if fmt[-1] in trim_list:
    #     fmt = fmt[:-1]

    times = []  # pd.TimeSeries or so..
    timestamps = []
    for p in paths:

        dt = time_from_filename_fast(p)
        timestamp = dt.replace(tzinfo=timezone.utc).timestamp()

        times.append(dt)
        timestamps.append(timestamp)

    return times, timestamps


def sort_by_number(paths,
                   strip_chars=STRIP_CHARS,
                   args=ARGS):
    """Parse a trailing number in the filename and sort paths."""
    n_files = len(paths)
    im_nums = np.ones(n_files) * np.nan

    for i in range(n_files):

        # Clean the filename of possible residual characters
        stem_clean = paths[i].stem.rstrip(strip_chars)
        # logging.debug(f"stem_clean: {stem_clean}")
        # Match a trailing number
        m = re.search(r'\d+$', stem_clean)

        # Store parsed number at correct index in the vector
        if m:
            # pure_name = stem_clean.split(m_str)[0]
            # pure_name = pure_name.rstrip(strip_chars)
            im_nums[i] = int(m.group())
            # logging.debug(f"match: {m.group()}")
            # logging.debug(f"num: {int(m.group())}")
        else:  # Just leaving the NaN
            logging.debug(f"Couldn't parse number from {paths[i]}")

    # logging.debug(f"im_nums: {im_nums}")

    # Sort image paths accordingly
    num_sort = np.argsort(im_nums)
    path_ar = np.array(paths)
    sorted_paths = path_ar[num_sort]
    # logging.debug(f"len sorted_paths: {len(sorted_paths)}")

    # Exclude files without number
    if not args.crazy and any(np.isnan(im_nums)):
        # Sort the image numbers
        sorted_nums = im_nums[num_sort]

        # Find index of first file to exclude from sorted files
        idx0_nan = np.argmax(np.isnan(sorted_nums))
        if idx0_nan == 0:
            logging.warning("No numbers parsed, excluding ALL files!")

        logging.debug(f"Excluding {n_files - idx0_nan} files")
        # Remove all NaN files
        sorted_paths = sorted_paths[:idx0_nan]
        logging.debug(f"{len(sorted_paths)} paths left")
    #     else:
    #         logging.warning("No NaN found!")

    return sorted_paths


def crop_by_time(paths, times, timestamps,
                 time_fmt=TIME_FMT_TRIMMED,
                 args=ARGS):
    """Interactively crop files according to dates."""
    dt0_orig = times[0]
    dtx_orig = times[-1]
    dt0 = times[0]
    dtx = times[-1]

    # Dialog 'Input date'
    done = False
    while not done:
        # dt0, dtx = dt0_orig, dtx_orig
        logging.info("\nEnter times in format 'yymmdd-HHMMSS'")
        t0_str = input(f"Start time [def={dt0}]: ")
        tx_str = input(f"  End time [def={dtx}]: ")
        try:
            dt0_mod = datetime.strptime(t0_str, time_fmt)
            if dt0_mod < dt0 or dt0_mod >= dtx:
                logging.warning("Entered impossible time, using default!")
                dt0_mod = dt0_orig
        except ValueError as err:
            logging.warning("Start time entered wrongly, using default!")
            dt0_mod = dt0_orig
        try:
            dtx_mod = datetime.strptime(tx_str, time_fmt)
            if dtx_mod <= dt0 or dtx_mod > dtx:
                logging.warning("Entered impossible time, using default!")
                dtx_mod = dtx_orig
        except ValueError as err:
            logging.warning("  End time entered wrongly, using default!")
            dtx_mod = dtx_orig
        logging.info("Including all images between " +
                     f"{dt0_mod} and {dtx_mod}")
        if not dt0_mod == dt0 or not dtx_mod == dtx:
            # change_times = True
            dt0 = dt0_mod
            dtx = dtx_mod
            logging.info("Manually changed interval to\n" +
                         f"Start time: {dt0}\n" +
                         f"  End time: {dtx}")
        q = input("Satisfied? [Y]/n: ")
        if not q.lower() == "n":
            done = True

    # Crop the files, times and timestamps.. akh pandas, i miss you!
    # Oki, let's go timestamps..
    ts0 = dt0.replace(tzinfo=timezone.utc).timestamp()
    tsx = dtx.replace(tzinfo=timezone.utc).timestamp()
    n_files = len(paths)

    # Iterate all timestamps.. akh numpy, i miss you!))
    # Find start index..
    idx0 = 0
    for ts in timestamps:
        if ts >= ts0:
            break
        idx0 += 1
    # Find stop index
    idxx = n_files - 1
    for ts in timestamps[::-1]:
        if ts <= tsx:
            break
        idxx -= 1

    # Crop paths, times, timestamps
    paths = paths[idx0:idxx + 1]
    times = times[idx0:idxx + 1]
    timestamps = timestamps[idx0:idxx + 1]
    logging.info(f"Cropped to {len(paths)} files (from {n_files})")

    dur_str = date_duration_string(times[0], times[-1])

    return paths, times, timestamps, dur_str


def crop_by_idx(paths, args=ARGS):
    """Interactively crop files according to index."""
    n_files = len(paths)
    idx0_orig = 0
    idxx_orig = n_files - 1
    idx0 = 0
    idxx = n_files - 1

    # Dialog 'Input index'
    done = False
    while not done:
        # idx0, idxx = idx0_orig, idxx_orig
        logging.info(f"\nEnter integers {idx0}-{idxx}")
        idx0_str = input(f"Start index [def={idx0}]: ")
        idxx_str = input(f"Final index [def={idxx}]: ")
        try:
            # idx0_mod = datetime.strptime(t0_str, time_fmt)
            idx0_mod = int(idx0_str)
            if idx0_mod < idx0 or idx0_mod >= idxx:
                logging.warning("Entered impossible index, using default!")
                idx0_mod = idx0_orig
        except ValueError as err:
            logging.warning("Start index entered wrongly, using default!")
            idx0_mod = idx0_orig
        try:
            # idxx_mod = datetime.strptime(tx_str, time_fmt)
            idxx_mod = int(idxx_str)
            if idxx_mod <= idx0 or idxx_mod > idxx:
                logging.warning("Entered impossible index, using default!")
                idxx_mod = idxx_orig
        except ValueError as err:
            logging.warning("  End index entered wrongly, using default!")
            idxx_mod = idxx_orig
        logging.info("Including all images between " +
                     f"{idx0_mod} and {idxx_mod}")

        if not idx0_mod == idx0 or not idxx_mod == idxx:
            # change_times = True
            idx0 = idx0_mod
            idxx = idxx_mod
            logging.info("Manually changed interval to\n" +
                         f"Start index: {idx0}\n" +
                         f"Final index: {idxx}")
        q = input("Satisfied? [Y]/n: ")
        if not q.lower() == "n":
            done = True

    # Crop paths
    paths = paths[idx0:idxx + 1]
    logging.info(f"Cropped to {len(paths)} files (from {n_files})")

    # Generate duration string for filename
    dur_str = f"img-{idx0}-{idxx}"

    return paths, dur_str


def reduce_files(step, paths, times=None, timestamps=None):
    """Filter out every STEP frame."""
    # Reduce the paths
    add_last = False
    paths_new = paths[::step]
    if paths_new[-1] != paths[-1]:
        add_last = True
        # If new last frame is not the absolute last frame
        paths_new.append(paths[-1])

    if not (times is None) and not (timestamps is None):
        if add_last:
            times = times[::step] + [times[-1]]
            timestamps = timestamps[::step] + [timestamps[-1]]
        else:
            times = times[::step]
            timestamps = timestamps[::step]

    return paths_new, times, timestamps


def check_interval(interval,
                   infer=False, timestamps=None,
                   framerate=ARGS.framerate,
                   multiply=ARGS.multiply,
                   add_last_sec=ADD_LAST_IMAGE_SECONDS,
                   def_interval=DEF_INTERVAL,
                   def_framerate=DEF_FRAMERATE,
                   def_multiply=DEF_MULTIPLY,
                   args=ARGS):
    """Compare inferred with passed intervals.

    Passed interval is in minutes,
    inferred interval is in seconds,
    returned interval is in seconds.
    """
    # Scale to seconds
    interval = interval * 60

    if infer and not (timestamps is None):
        # Infer the interval between pictures in minutes
        ts_a = np.array(timestamps)
        delta_t_sec = (ts_a[1:] - ts_a[:-1])  # / 60.0
        inferred = round(median(delta_t_sec))

        # Compare inferred interval to passed or default
        if inferred != interval:
            logging.warning(f"Overwriting interval {interval} with " +
                            f"inferred interval {inferred}")
            interval = inferred

    # Make sure it's not less than a second
    if interval < 1:
        logging.warning(f"Interval {interval} < 1, setting to 1")
        interval = 1
    # Ensure integer
    interval = round(interval)

    # Check framerate
    #
    # if args.framerate:
    #     fr_prompt = "Framerate [def={}]: ".format(def_framerate)
    #     framerate = input(fr_prompt)
    #     try:
    #         framerate = int(framerate)
    #         if framerate < 2 or framerate > 90:
    #             logging.warning(("Framerate {} will be changed " +
    #                              "to {} [default]").format(
    #                                  framerate, def_framerate))
    #             framerate = def_framerate
    #     except Exception as err:
    #         logging.warning(("Framerate {} will be changed " +
    #                          "to {} [default], error: {}").format(
    #                              framerate, def_framerate, err))
    #         framerate = def_framerate
    # else:
    #     framerate = def_framerate
    # logging.info("Framerate set to {}".format(framerate))
    #
    try:
        if framerate < 10:
            logging.warning(f"Framerate {framerate} < 10, setting to 10")
            framerate = 10
        if framerate > 100:
            logging.warning(f"Framerate {framerate} > 100, setting to 100")
            framerate = 100
    except TypeError as err:
        logging.warning(f"Framerate '{framerate}' has wrong " +
                        f"type '{type(framerate)}', replacing with " +
                        f"default framerate {def_framerate}. " +
                        f"Error: {err}")
        framerate = def_framerate
    framerate = round(framerate)

    # Check frame reduction or multiplication
    # Only ONE of the two can be set! Prefer multiply?
    # factor_lock = False
    if multiply > 1:
        # factor_lock = True
        # Multiplying frames reduces the number of timesteps per second
        multiplicator = 1 / float(multiply)
        logging.info(f"Add each frame {multiply} times.")
        # NOTE: 'reduce' has already been performed before
        # elif reduce > 1:
        #     multiplicator = float(reduce)
        #     logging.info(f"Delete frame every {reduce} frames.")
    else:
        multiplicator = 1.0

    # Calculate minutes of real time per second of video
    # TODO: STill use passed interval, even if no times available
    sps = (framerate * interval * multiplicator)  # / 60.0
    if infer:
        if sps < 60:
            speed_str = f"{int(round(sps))}sps"
        else:
            speed_str = f"{int(round(sps / 60.0))}mps"
    else:
        speed_str = f"{int(round(framerate))}fps"
    logging.debug(f"Speed string: {speed_str}")

    # Compute number of frames to add
    add_last_frames = framerate * add_last_sec

    return interval, framerate, add_last_frames, speed_str


def propose_name(name, duration, speed, dir_out,
                 prefix=VID_PREFIX,
                 ext=DEF_VID_EXT,
                 args=ARGS):
    """Generate a template name for the output video."""
    # Propose prefix
    # n0 = os.path.splitext(ffn0.split("/")[-1])[0]

    suggestion = f"{prefix}{name}_{duration}_{speed}{ext}"

    # Dialog 'Input index'
    if args.interactive or args.name:
        logging.info(f"Proposed video-name: {suggestion}")
        sugg_orig = suggestion
        name_orig = name
        done = False
        while not done:

            logging.info(f"Name: {name}, filename: {suggestion}")
            q = input("[A]ccept (DEFAULT), change [n]ame, " +
                      "or [c]hange filename: ")
            if q.lower().startswith("a") or q == "":
                done = True
            elif q.lower().startswith("n"):
                name = input(f"New name (default: {name_orig}): ")
                if not name:
                    name = name_orig
                suggestion = f"{prefix}{name}_{duration}_{speed}{ext}"
            elif q.lower().startswith("c"):
                fn = input("New filename without ext " +
                           f"(default: {sugg_orig}): ")
                if not fn:
                    suggestion = sugg_orig
                else:
                    suggestion = fn + ext
    logging.info(f"Selected filename: {suggestion}")

    # Assemble Path object
    filepath = ioh.safename(dir_out / suggestion, "file")
    logging.debug(f"Full final path: {filepath}")

    return filepath


def parse_expinfo(filename):
    """Read Hive and RPi number from filename.

    Filename e.g.:
    bgx_hive1_rpi1_190727-140003-utc.jpg
    """
    hive = int(filename.split("hive")[-1][0])
    rpi = int(filename.split("rpi")[-1][0])

    return hive, rpi


def label_frame(frame, filename, dt_utc,
                time_fmt=TIMESTAMP_FMT, args=ARGS):
    """Create a textbox with timestamp and infos."""
    # logging.debug("frame.shape: {}".format(frame.shape))
    height, width, chans = frame.shape
    rect_height = 35
    rect_width = 400

    # # Calculate timestamp
    # sec = frame_i / float(fps)
    # min = int(sec / 60)
    # sec = int(sec % 60)

    # NOTE: Broodnest experiment-specific!
    hive, rpi = parse_expinfo(filename)

    # Make label
    # text = "exp{:02} - {:02}:{:02}".format(exp, min, sec)
    text = f"Hive{hive} RPi{rpi} {dt_utc.strftime(time_fmt)}"
    # logging.debug("Putting text: {}".format(text))

    # Set rectangle coordinates
    # rect_pt1 = (height - rect_height, width - rect_width)
    # rect_pt2 = (height, width)
    rect_pt1 = (width - rect_width, height - rect_height)
    rect_pt2 = (width, height)

    # Set text coordinates
    # text_origin = (height - 1, rect_pt1[1] + 1)
    text_origin = (rect_pt1[0] + 5, height - 10)

    # Set colors
    text_color = (255, 255, 255)
    rect_color = (0, 0, 0)
    rect_thickness = cv2.FILLED
    # Draw rectangle
    cv2.rectangle(frame, rect_pt1, rect_pt2,
                  rect_color, rect_thickness)
    # logging.debug("Placed rectangle at ({}, {})".format(
    #     rect_pt1, rect_pt2))

    # Write text
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontsize = 0.7
    text_thickness = 1
    linetype = cv2.LINE_AA  # LINE_8
    cv2.putText(frame, text, text_origin, font, fontsize,
                text_color, text_thickness, linetype)
    # logging.debug("Placed text at {}".format(text_origin))

    return frame


def log_info(frame_count, n_frames, t0, args=ARGS):
    """Calculate progress and output to screen and logfile."""
    # Take time
    ti = time.time()
    dt_s = ti - t0
    dt_mins = int(dt_s // 60)
    dt_secs = int(dt_s % 60)

    # Get ratio of done frames
    done = frame_count / float(n_frames)
    proc_fps = frame_count / float(dt_s)

    # Predict finishing time
    if done > 0:
        pred_tx = dt_s / float(done)
    else:
        pred_tx = 0.0
    tx_mins = int(pred_tx // 60)
    tx_secs = int(pred_tx % 60)
    t_left = pred_tx - dt_s

    # Print it all to screen and logfile
    logging.info(("{:6}% done ({:5}/{:5}) " +
                  "took {:02}:{:02} min at {:5} fps, " +
                  "predicted end: {:02}:{:02} " +
                  "({} seconds)").format(
        round(100 * done, 2),
        int(frame_count), int(n_frames),
        dt_mins, dt_secs, round(proc_fps, 2),
        tx_mins, tx_secs, round(t_left, 2)))


def main(
        vid_prefix=VID_PREFIX,
        time_fmt=TIME_FMT,
        quality=QUALITY,
        add_final_imgs=ADD_LAST_IMAGE_SECONDS,
        info_skip=SKIP_FRAMES_INFO,
        min_bytes=MINIMAL_FILESIZE,
        args=ARGS,
        ):
    """Make a video from images."""
    # global OUT_APPEND

    # Set up IO-environment and logging
    dir_in, dir_out = initialize_io()

    # Check the image pattern
    img_pattern, ext = correct_img_pattern(args.pattern)

    # Find files
    # ffn_list = ioh.parse_subtree(dir_in, img_pattern)
    img_paths = sorted(dir_in.rglob(img_pattern))
    n_imgs = len(img_paths)
    logging.info(f"Found {n_imgs} matches for " +
                 f"'{img_pattern}' in {dir_in}")
    first_img = img_paths[0]
    last_img = img_paths[-1]
    logging.info("\n" +
                 f"\t First file: {first_img}\n" +
                 f"\t  Last file: {last_img}")

    # Try parsing timestamps and check on the interval
    dt_obj, stem = time_from_filename_cautious(first_img)
    if dt_obj:
        times, timestamps = get_time_info(img_paths)
        got_times = True
        logging.info("Successfully parsed times!\n" +
                     f"Found images from {times[0]} to {times[-1]}")
        # Generate duration string for filename
        # dur_str = f"{times[0].strftime(time_fmt)}-"
        dur_str = date_duration_string(times[0], times[-1])
    else:
        img_paths = sort_by_number(img_paths)
        got_times = False
        times = None
        timestamps = None
        logging.info("Found no timestamps in filenames")
        dur_str = f"img-0-{n_imgs}"
    logging.debug(f"Duration string: {dur_str}")

    # Check which ones to exclude (based on times or just the numbers)
    # Interactively trim filelist according to date
    if args.interactive and not args.all:
        if got_times:
            img_paths, times, timestamps, dur_str = crop_by_time(
                img_paths, times, timestamps)
        else:
            img_paths, dur_str = crop_by_idx(img_paths)

    # Delete or multiply regularly spaced frames

    assert not ((args.multiply > 1) and (args.reduce > 1)), (
        "Both multiply and reduce are set!")
    if args.reduce > 1:
        img_paths, times, timestamps = reduce_files(
            args.reduce,
            img_paths, times, timestamps)

    # Retrieve the interval, framerate and a timespeed string
    # TODO: Do this after all the cropping?
    interval, framerate, add_last, speed_str = check_interval(
        args.interval, infer=got_times, timestamps=timestamps)

    # Generate full path to the output video
    vid_path = propose_name(stem, dur_str, speed_str, dir_out)

    # Calculate final frame number
    n_frames = len(img_paths) * args.multiply + add_last
    logging.info(f"Writing {n_frames} frames in total.")

    # Open first image and read properties
    # Flags for cv2.imread():
    # https://docs.opencv.org/3.4.2/d4/da8/
    # group__imgcodecs.html#ga61d9b0126a3e57d9277ac48327799c80
    img = cv2.imread(str(first_img))
    try:
        resolution = (img.shape[1], img.shape[0])
    except AttributeError as err:
        logging.error(f"Couldn't read first image from {first_img}. " +
                      f"Error: {err}\n"
                      "leaving program...")
        return None
    except IndexError as err:
        logging.error(f"Couldn't read first image from {first_img}. " +
                      f"Error: {err}\n"
                      "leaving program...")
        # TODO: Implement a while loop here?
        return None

    # Create VideoWriter object
    # https://docs.opencv.org/3.4.2/dd/d43/tutorial_py_video_display.html
    # Define the codec
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    outmov = cv2.VideoWriter(str(vid_path), fourcc,
                             float(framerate), resolution)

    # Iterate over all remaining files
    failed_ims = []
    t0 = time.time()
    try:
        frame_count = 0
        img_idx = 0
        for p in img_paths:

            # Check filesize
            filesize = p.stat().st_size
            if filesize >= min_bytes:
                try:
                    # Read image
                    img = cv2.imread(str(p))

                    # Make timestamp box
                    if args.timestamp and got_times:
                        dt = times[img_idx]
                        img_idx += 1
                        # ts = timestamps[img_idx]

                        img = label_frame(img, p.name, dt)

                    # Write image to movie
                    for i in range(args.multiply):
                        # https://stackoverflow.com/questions/11337499/
                        # how-to-convert-an-image-from-np-uint16-to-np-uint8
                        outmov.write(img)
                        # outmov.write(np.uint8(img))
                        frame_count += 1

                        if frame_count % info_skip == 0:
                            log_info(frame_count, n_frames, t0)
                            logging.debug(f"{p.name}")

                except Exception as err:
                    logging.error(f"Failed to read image from {p}, " +
                                  f"error: {err}")
                    failed_ims.append(["error", p])
            else:
                logging.error(f"Failed to read image from {p}, " +
                              f"filesize {filesize} < {min_bytes} bytes")
                failed_ims.append(["size", p])

    finally:
        outmov.release()
        cv2.destroyAllWindows()
        logging.info(f"Closed movie, last visited file: {p}")

    # Display failed image files
    n_fails = len(failed_ims)
    if n_fails > 0:
        logging.warning(f"{n_fails} files were not processed:")

        formatted = "\n"
        for info in failed_ims:
            formatted += f"{info[0]}:\t{info[1]}\n"
        logging.info(formatted)

    # Display the video location
    logging.info(f"Done! Find your fresh video at {vid_path}")


if __name__ == "__main__":
    main()
