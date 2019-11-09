#!/usr/bin/env python3
"""Repair wrongly named files.

Filenames are e.g.:
./csv/hive2/rpi1/bgx_hive2_rpi1_targ190907-14_190907-140003-utc.csv

and should be:
./csv/hive1/rpi2/bgx_hive1_rpi2_targ190907-14_190907-140003-utc.csv

i.e. switch hive2 and rpi1 -> hive1 and rpi2

and move them to the correct folder, i.e.:
./csv/hive/rpi2/
"""
# import os
# import shutil
from pathlib import Path

# import pytz

PATH_IN = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
    "broodnest_bgs/assemble_paths_191108-utc/csv/hive2"
)
PATH_OUT = Path(
    "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
    "broodnest_bgs/assemble_paths_191108-utc/csv/hive1/rpi2"
)

# LOCAL_TZ = pytz.timezone("Etc/UTC")
# LOCAL_TZ = pytz.timezone("Europe/Vienna")
# TIME_FMT = "%y%m%d-%H%M%S-utc"
# DAY_FMT = "day-%y%m%d"
# WRONG_FMT = "%d_%m_%H_%M_%S.jpg"
PREFIX = "bgx_hive1_rpi2_targ"
EXIST_TAG = "_exists"


def main(path_in=PATH_IN, path_out=PATH_OUT,
         prefix=PREFIX, exists=EXIST_TAG):
    """Iterate over all broodnest photos and rename them."""
    # Iterate over all images

    # ./hive2/rpi1/bgx_hive2_rpi1_targ190907-14_190907-140003-utc.csv
    filelist = sorted(path_in.rglob("bgx_hive2_rpi1_targ*.csv.jpg"))
    print(f"Found {len(filelist)} files.")
    for file in filelist:

        # Parse the correct file ending
        timetail = file.name.split("targ")[-1]

        # Build the Path with the correct filename
        filepath = path_out / (prefix + timetail)

        # # Create outfolder if necessary
        # if not outfolder.is_dir():
        #     outfolder.mkdir(parents=True)
        #     print(f"Created folder '{outfolder}'")

        # Check if file exists
        if filepath.is_file():
            print(f"WARNING: File '{filepath}' already exists!")
            filepath = path_out / (prefix + timetail + exists)
            print(f"Renamed file to '{filepath.name}'")

        # Move the file
        file.replace(filepath)
        print(f"Moved file to '{file}'")


if __name__ == "__main__":
    main()  # (args)
