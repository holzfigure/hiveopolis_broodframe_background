#!/usr/bin/env python3
"""Rename files such that timestamps are in UTC and sort correctly."""
# import os
import shutil
from pathlib import Path
from datetime import datetime

import pytz

PATH_IN = Path("/media/holzfigure/Data/NAS/NAS_incoming_data")
PATH_OUT = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
    "broodnest_obs/hive1"
)

# Filename e.g.:  pi1_hive1broodn_15_8_0_0_4.jpg
INFILE_PATTERN = "pi*_hive*broodn_*.jpg"
# Foldername e.g.:  Photos_of_Pi1_1_9_2019
# Foldername e.g.:  Photos_of_Pi1_heating_1_11_2019
INFOLDER_PATTERN = "Photos_of_Pi*/"

OUTRAW_PREFIX = "raw"


YEAR = 2019
# LOCAL_TZ = pytz.timezone("Etc/UTC")
LOCAL_TZ = pytz.timezone("Europe/Vienna")
TIME_FMT = "%y%m%d-%H%M%S-utc"
DAY_FMT = "day-%y%m%d"
TIME_INFILE_SPLIT_TSTR = "broodn_"
TIME_INFILE_FMT = "%d_%m_%H_%M_%S.jpg"
TIME_INFOLDER_FMT = "%d_%m_%Y"


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
    dt_utc = convert2utc(dt_naive)

    return dt_utc


def get_utc_timestrings(fn, year=YEAR, local_tz=LOCAL_TZ,
                        time_fmt=TIME_FMT, day_fmt=DAY_FMT,
                        wrong_fmt=TIME_INFILE_FMT):
    """Parse and localize the time, output new timestrings."""
    # Extract the timestring from the filename
    t_str = fn.split("broodn_")[-1]

    # Parse it into a datetime object
    dt_naive = datetime.strptime(
            t_str, wrong_fmt).replace(year=year)

    # Localize and convert to UTC
    dt_local = local_tz.localize(dt_naive)
    dt_utc = dt_local.astimezone(pytz.utc)

    # Generate the new timestrings
    t_str = dt_utc.strftime(time_fmt)
    day_str = dt_utc.strftime(day_fmt)

    return t_str, day_str


def main(path_in=PATH_IN, path_out=PATH_OUT,
         file_pattern=INFILE_PATTERN):
    """Iterate over all broodnest photos and rename them."""
    # Iterate over all images
    n = 0
    for file in path_in.rglob("pi*_hive1broodn_*.jpg"):
        n += 1

        # Get nice UTC timestrings
        t_str, day_str = get_utc_timestrings(file.name)

        # Parse which RPi took photo
        rpi_num = int(file.name.split("_hive*broodn")[0][-1])

        # Get name of output folder
        outfolder = (path_out / f"rpi{rpi_num}" /
                     f"hive1_rpi{rpi_num}_{day_str}")

        # Create outfolder if necessary
        if not outfolder.is_dir():
            outfolder.mkdir(parents=True)
            print(f"Created folder '{outfolder}'")

        # Get name of the output file
        outfile = outfolder / f"hive1_rpi{rpi_num}_{t_str}.jpg"

        # Copy the file (while attempting to keep metadata)
        shutil.copy2(file, outfile)

        if n % 1000 == 0:
            print(f"Handled {n} files for now..")


if __name__ == "__main__":
    main()  # (args)
