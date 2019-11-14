#!/usr/bin/env python3
"""Rename files such that timestamps are in UTC and sort correctly."""
import os
import glob
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# import pytz

# PATH_IN = Path("/media/holzfigure/Data/NAS/NAS_incoming_data")
# PATH_OUT = Path(
#     "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/"
#     "broodnest_obs"
# )

# PATH_IN = Path(
#     "/media/holzfigure/Data/local_stuff/Hiveopolis/"
#     "broodnests/sample_raw_folders"
# )
# PATH_OUT = Path(
#     "/media/holzfigure/Data/local_stuff/Hiveopolis/"
#     "broodnests/new_scheme"
# )

PATH_IN = Path("/volume1/incoming_data")
PATH_OUT = Path("/volume1/incoming_data/Hiveopolis/broodnest_obs")

PATH_ERR = Path(PATH_OUT / "DST_troubles")


# Filename e.g.:  pi1_hive1broodn_15_8_0_0_4.jpg
INFILE_PATTERN = "pi*_hive*broodn_*.jpg"
# Foldername e.g.:  Photos_of_Pi1_1_9_2019
# Foldername e.g.:  Photos_of_Pi1_heating_1_11_2019
INFOLDER_PATTERN = "Photos_of_Pi*/"

OUTRAW_PREFIX = "raw"


YEAR = 2019
# Transition from daylight saving time took place on Oct 27 2019.
# At 3 am CEST, time switched to 2 am CET
DST_DT1 = datetime(2019, 10, 27, 2)
DST_DT2 = datetime(2019, 10, 27, 3)
UTC_OFFSET = timedelta(hours=1)
# LOCAL_TZ = pytz.timezone("Etc/UTC")
# LOCAL_TZ = pytz.timezone("Europe/Vienna")
TIME_FMT = "%y%m%d-%H%M%S-utc"
DAY_FMT = "day-%y%m%d"
TIME_INFILE_SPLIT_TSTR = "broodn_"
TIME_INFILE_FMT = "%d_%m_%H_%M_%S.jpg"
TIME_INFOLDER_FMT = "%d_%m_%Y"


# def convert2utc(dt_naive, local_tz=LOCAL_TZ):
#     """Localize naive time and convert it to an aware datetime in UTC.
#
#     LOCAL_TZ = pytz.timezone("Etc/UTC")
#     LOCAL_TZ = pytz.timezone("Europe/Vienna")
#     """
#     # Localize (i.e., make aware)
#     dt_local = local_tz.localize(dt_naive)
#     # Convert to UTC
#     dt_utc = dt_local.astimezone(pytz.utc)
#
#     return dt_utc


def convert2utc_naive(dt_naive, utc_os=UTC_OFFSET,
                      dst_dt1=DST_DT1, dst_dt2=DST_DT2):
    """Change naive datetime object to UTC in a hacky way.

    Only works for 2019 and not the day of the Oct DST_OFF transition.

    Transition from daylight saving time took place on Oct 27 2019.
    At 3 am CEST, time switched to 2 am CET
    """
    # # Localize (i.e., make aware)
    # dt_local = local_tz.localize(dt_naive)
    # # Convert to UTC
    # dt_utc = dt_local.astimezone(pytz.utc)

    # dt_naive.replace(hour=0, minute=0, second=0, microsecond=0)
    if dt_naive < dst_dt1:
        # DST ON -> CEST (minus 2 hours)
        dt_naive_utc = dt_naive - utc_os - timedelta(hours=1)
    elif dt_naive > dst_dt1:
        # DST OFF -> CET (minus 1 hour)
        dt_naive_utc = dt_naive - utc_os
    else:
        print(
            "WARNING: Found file from potentially within the "
            # f"DST transition: dt_naive = {dt_naive}"
            "DST transition: dt_naive = {}".format(dt_naive)
        )
        dt_naive_utc = None

    return dt_naive_utc


# def folder_datetime(foldername, time_infolder_fmt=TIME_INFOLDER_FMT):
#     """Parse UTC datetime from foldername.
#
#     Foldername e.g.:  Photos_of_Pi1_1_9_2019/
#                       Photos_of_Pi2_heating_1_11_2019/
#     """
#     # t_str = folder.name.split("Photos_of_Pi")[-1][2:]  # heating!!
#     t_str = "_".join(foldername.split("_")[-3:])
#     day_naive = datetime.strptime(t_str, time_infolder_fmt)
#     # # Localize as UTC
#     # # day_local = local_tz.localize(day_naive)
#     # # dt_utc = day_local.astimezone(pytz.utc)
#     # day = pytz.utc.localize(day_naive)
#
#     return day_naive


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
    dt_utc = convert2utc_naive(dt_naive)

    return dt_utc


def get_utc_timestrings(fn, year=YEAR,  # local_tz=LOCAL_TZ,
                        time_fmt=TIME_FMT, day_fmt=DAY_FMT,
                        wrong_fmt=TIME_INFILE_FMT):
    """Parse and localize the time, output new timestrings."""
    # Extract the timestring from the filename
    t_str = fn.split("broodn_")[-1]

    # Parse it into a datetime object
    dt_naive = datetime.strptime(
            t_str, wrong_fmt).replace(year=year)

    # Localize and convert to UTC
    dt_utc = convert2utc_naive(dt_naive)

    if dt_utc is not None:
        # Generate the new timestrings
        t_str = dt_utc.strftime(time_fmt)
        day_str = dt_utc.strftime(day_fmt)
    else:
        # Within possibly DST transition..
        t_str = None
        day_str = None

    return t_str, day_str


def main(path_in=PATH_IN, path_out=PATH_OUT, path_err=PATH_ERR,
         file_pattern=INFILE_PATTERN,
         folder_pattern=INFOLDER_PATTERN):
    """Iterate over all broodnest photos and rename them."""
    # Iterate over all images
    n = 0
    # # for file in path_in.rglob(file_pattern):
    # # Fails in NAS because of ../incoming_data/#recycle/.. folders!
    # folders = sorted(path_in.glob(folder_pattern),
    #                  key=os.path.getmtime)  # , reverse=True)
    # # Akh os doesn't take pathlib Paths in Python 3.5..
    folders = sorted(glob.iglob(str(path_in) + '/' + folder_pattern),
                     key=os.path.getmtime)  # , reverse=True)
    # n_folders = len(folders)
    print("Number of folders: {}".format(len(folders)))
    for folder in folders:
        # folder = Path(folder)
        print("Processing folder '{}'".format(folder))

        files = sorted(glob.iglob(str(folder) + '/' + file_pattern),
                       key=os.path.getmtime)  # , reverse=True)
        print("Folder contains {} matching files".format(len(files)))

        for file in files:
            file = Path(file)
            n += 1

            # Get nice UTC timestrings
            t_str, day_str = get_utc_timestrings(file.name)

            if t_str is not None:
                # Parse Hve and RPi took photo
                filename = file.name
                trunc = filename.split("broodn")[0]
                rpi = int(trunc.split("pi")[-1][0])
                hive = int(trunc.split("hive")[-1][0])
                # print(f"Filename: {filename}, rpi={rpi}, hive={hive}")
                # print("Filename: {}, rpi={}, hive={}".format(
                #     filename, rpi, hive))

                # NOTE: Temporary hack to fix wrongly named hive2 files
                # TODO: REMOVE, especially when "hive2" really exists!
                if hive == 2:
                    hive = 1
                    rpi = 2
                    print(
                        "WARNING: Changed Hive and RPi numbers: "
                        # f"Filename: {filename}, rpi={rpi}, hive={hive}"
                        "Filename: {}, rpi={}, hive={}".format(
                            filename, rpi, hive)
                    )
                # Get name of output folder
                # outfolder = (path_out / f"rpi{rpi_num}" /
                #              f"hive1_rpi{rpi_num}_{day_str}")
                outfolder = (
                        path_out /
                        ("hive" + str(hive)) /
                        ("rpi" + str(rpi)) /
                        "hive{}_rpi{}_{}".format(hive, rpi, day_str)
                )

                # Create outfolder if necessary
                if not outfolder.is_dir():
                    outfolder.mkdir(parents=True)
                    # print(f"Created folder '{outfolder}'")
                    print("Created folder '{}'".format(outfolder))

                # Get name of the output file
                # outfile = outfolder / f"hive1_rpi{rpi}_{t_str}.jpg"
                outfile = outfolder / "hive{}_rpi{}_{}.jpg".format(
                    hive, rpi, t_str)

            else:
                print(
                    "ERROR: Found file possibly in DST-transition: "
                    "{}".format(file)
                )
                outpath = path_err / file.parent.name
                if not outpath.is_dir():
                    outpath.mkdir(parents=True)
                    print("Created folder '{}'".format(outpath))
                outfile = outpath / filename

            # Copy the file (while attempting to keep metadata)
            if outfile.is_file():
                print("WARNING: File '{}' exists!".format(outfile))
                outfile = outfile.with_name(outfile.name + "_DUPLICATE")
                print("WARNING: Renamed to '{}'".format(outfile))

            # try:
            shutil.copy2(str(file), str(outfile))
            # print("Copied {} to {}".format(file, outfile))
            # except IsADirectoryError as err:
            #     print("IsADirectoryError: {}".format(err))
            #     outfile =

            if n % 1000 == 0:
                # print(f"Handled {n} files for now..")
                print("Handled {} files for now..".format(n))


if __name__ == "__main__":
    main()  # (args)
