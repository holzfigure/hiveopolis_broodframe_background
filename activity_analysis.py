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
import os
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
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection

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
    # "/media/holzfigure/Data/local_stuff/Hiveopolis/broodnests/"
    # "bee_activity_csvs/csv/"  # hive1/rpi1/"
    "F:/Hiveopolis/comb_activity/bee_activity_191125-utc/csv"
)
# PATH_OUT = Path(
#     "/media/holzfigure/Data/NAS/NAS_incoming_data/Hiveopolis/" +
#     "broodnest_activity/csv"
# )
PATH_OUT = Path(
    # "/media/holzfigure/Data/local_stuff/Hiveopolis/broodnests/"
    # "bee_activity"
    "F:/Hiveopolis/comb_activity"
)
# Filename e.g.: "act_hive1_rpi1_190804_000000-235959-utc_euclidean.csv"
INFILE_PATTERN = "act_hive*_rpi*-utc*.csv"
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

# Plotting Parameters
# e.g.              147237858
OUTLIER_THRESHOLD = 165000000  # e.g. 147237858
# Plot properties
# RESOLUTION = (19.20, 10.80)
# RESOLUTION = (7.2, 4.8)
# RESOLUTION = (9.6, 5.4)
# RESOLUTION = (16.0, 9.0)
RESOLUTION = (12.8, 7.2)
RESOLUTION2 = (10.8, 8.8)

# LINESTYLES = ['-', '--', '-.', ':']
# COLORMAP_NAME = "gist_ncar"  # "viridis"
DEF_EXT = "png"
TITLE_FS = 20
AXIS_FS = 19
TICK_FS = 16
LEGEND_FS = 16
N_BINS = "auto"

LEVEL_OF_SIGNIFICANCE = 0.05
DEF_SAMPLING_INTERVAL = 1
# SAMPLING_INTERVAL = 100  # = 20 seconds (100 values = 0.2 * 5 * 20)

# Argument parsing
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
    ioh.setup_logging(thisname, args, dir_log=dir_out / "log")
    dir_out = Path(dir_out)
    logging.info(f"input from {dir_in}, output to {dir_out}")

    # Setting up MAX_THREADS for numExpr  (12 on Hiveopolis station in BeeLab)
    # NOTE: Do only on HO beelab station PC!!
    # https://buildmedia.readthedocs.org/media/pdf/numexpr/latest/numexpr.pdf
    # https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/api.html
    os.environ['NUMEXPR_MAX_THREADS'] = '12'

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
    day_str = parts[3]
    method = parts[5]

    # Parse Hive and RPi number
    hive = int(hive_str[-1])
    rpi = int(rpi_str[-1])
    method = method.strip(".csv")

    # # Parse timestring into a datetime object
    # dt_naive = datetime.strptime(t_str, time_fmt)
    # dt_utc = pytz.utc.localize(dt_naive)

    return hive, rpi, method, day_str


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Taken from:
    https://stackoverflow.com/a/50029441/8511824

    Example usage:

    xs = [[0, 1], [0, 1, 2]]
    ys = [[0, 0], [1, 2, 1]]
    c = [0, 1]
    lc = multiline(xs, ys, c, cmap='bwr', lw=2)

    Or:

    n_lines = 30
    x = np.arange(100)

    yint = np.arange(0, n_lines * 10, 10)
    ys = np.array([x + b for b in yint])
    xs = np.array([x for i in range(n_lines)])  # could also use np.tile

    colors = np.arange(n_lines)

    fig, ax = plt.subplots()
    lc = multiline(xs, ys, yint, cmap='bwr', lw=2)

    axcb = fig.colorbar(lc)
    axcb.set_label('Y-intercept')
    ax.set_title('Line Collection with mapped colors')


    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for
                      each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """
    # Find axes
    ax = plt.gca() if ax is None else ax

    # Create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # Set coloring of line segments
    # NOTE: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # Add lines to axes and rescale
    # NOTE: adding a collection doesn't autoscale xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def plot_single_activity(
        series, name, path_out,
        title_fs=TITLE_FS,
        axis_fs=AXIS_FS,
        tick_fs=TICK_FS,
        legend_fs=LEGEND_FS,
        resolution=RESOLUTION,
        args=ARGS,
):
    """Plot a single activity curve and save the image.

    https://jakevdp.github.io/PythonDataScienceHandbook/
    03.11-working-with-time-series.html

    see also "df.rolling()" and "pd.rolling_mean()"
    """
    fig, ax = plt.subplots(figsize=resolution, dpi=100)
    series.plot(alpha=0.3, color="blue", style="-", ax=ax)

    # series.resample('h').mean().plot(
    #         label="mean", style='-', color="black", linewidth=2, ax=ax)

    h_median = series.resample('h').median()
    h_median.plot(
            label="median", style='-', color="red", linewidth=2, ax=ax)

    # lc = multiline(xs, ys, yint, cmap='bwr', lw=2)
    #
    # axcb = fig.colorbar(lc)
    # axcb.set_label('Y-intercept')
    # ax.set_title('Line Collection with mapped colors')

    # if args.legend:
    #     axes[0].legend(  # ncol=1, # borderaxespad=0.,
    #         borderaxespad=0.,
    #         loc="upper right",  # bbox_to_anchor=(1.1, 1.0),
    #         fontsize=leg_fs,
    #         fancybox=True, framealpha=0.5)
    # # plt.legend(ncol=2, borderaxespad=0.)
    ax.legend()

    # ffn = ioh.safename(path_out / f"{name}.png", "file")
    ffn = path_out / f"{name.lower()}.png"
    plot_path = ioh.safesavefig(ffn)

    logging.debug(f"Figure exported to {plot_path}")

    return plot_path, h_median


def hourly_bxpl_single(
        df, name, path_out,
        title_fs=TITLE_FS,
        axis_fs=AXIS_FS,
        tick_fs=TICK_FS,
        legend_fs=LEGEND_FS,
        resolution=RESOLUTION,
        args=ARGS,
):
    """Make a boxplot grouped by hour.

    df ... datetime-indexed bee activity

    df['date_of_birth'].map(lambda d: d.month).plot(kind='hist')

    https://pandas.pydata.org/pandas-docs/stable/user_guide/
    visualization.html#visualization-box

    To group by arbitrary time, use pd.TimeGrouper():
    https://stackoverflow.com/questions/34814606/
    a-per-hour-histogram-of-datetime-using-pandas/34820891
    """

    # Group data by hour
    # hourly = series.index.map(lambda d: d.hour)
    # hourly = series.groupby(series.index.hour)
    # logging.debug(f"hourly: {hourly}, list: {list_hourly}")

    fig, ax = plt.subplots(figsize=resolution, dpi=100)
    # series.plot(kind="box", ax=ax)
    df.boxplot(column="activity", by="hour", ax=ax)

    # ffn = ioh.safename(path_out / f"{name}.png", "file")
    ffn = path_out / f"{name.lower()}_bp.png"
    plot_path = ioh.safesavefig(ffn)

    logging.debug(f"Figure exported to {plot_path}")

    return plot_path


def plot_median_days(
        med_list, name, path_out,
        title_fs=TITLE_FS,
        axis_fs=AXIS_FS,
        tick_fs=TICK_FS,
        legend_fs=LEGEND_FS,
        resolution=RESOLUTION,
        args=ARGS,
):
    """Plot all median day curves with color depending on the date.

    https://stackoverflow.com/questions/38208700/
    matplotlib-plot-lines-with-colors-through-colormap
    """
    n_lines = len(med_list)

    fig, ax = plt.subplots(figsize=resolution, dpi=100)

    # Get the colors for the lines
    # TODO: Check for unique days and have the same day consistent..
    # TODO: Make colors really time-dependent
    lc_idx = np.linspace(0, 1, n_lines)
    colors = plt.cm.viridis(lc_idx)

    # Plot all curves
    for i in range(n_lines):
        # NOTE: Using "color" instead of "c" throws an error!
        med_list[i].plot(ax=ax, c=colors[i])

    # lc = multiline(ax=ax)
    # lc = multiline(xs, ys, c, cmap='bwr', lw=2)
    # lc = multiline(xs, ys, yint, cmap='bwr', lw=2)
    #
    # axcb = fig.colorbar(lc)
    # axcb.set_label('Y-intercept')
    # ax.set_title('Line Collection with mapped colors')

    # if args.legend:
    #     axes[0].legend(  # ncol=1, # borderaxespad=0.,
    #         borderaxespad=0.,
    #         loc="upper right",  # bbox_to_anchor=(1.1, 1.0),
    #         fontsize=leg_fs,
    #         fancybox=True, framealpha=0.5)
    # # plt.legend(ncol=2, borderaxespad=0.)
    # ax.legend()

    # ffn = ioh.safename(path_out / f"{name}.png", "file")
    ffn = path_out / f"{name.lower()}_medians"
    plot_path = ioh.safesavefig(ffn)
    logging.debug(f"Figure exported to {plot_path}")

    # Cast all medians to the same day
    datelist = []
    for h_median in med_list:
        datelist.extend(h_median.index)

    # datelist = sorted(datelist)
    # mean_date = np.mean(datelist)
    # mean_date = pd.to_timedelta(datelist).mean()
    # datelist.resample('5Min').
    mean_date = pd.to_datetime(datelist).mean()
    m_year = mean_date.year
    m_month = mean_date.month
    m_day = mean_date.day
    logging.info(f"Parsed average date: {mean_date}")

    fig, ax = plt.subplots(figsize=resolution, dpi=100)

    sd_list = []
    # xs = []
    # ys = []
    for i in range(n_lines):
        h_median = med_list[i]
        # sd_median = pd.to_datetime(h_median.index).dt.replace(
        #         year=m_year, month=m_month, day=m_day)

        # Get the day of the line
        # Pick some point in the middle of the line (to avoid day-change)
        idx = int(len(h_median.index) / 2)
        dt = h_median.index[idx]
        daylabel = dt.strftime("%y-%m-%d")

        dtseries = pd.Series(h_median.index)
        # sd_times = vec_dt_replace(
        #         dtseries,
        #         m_year, m_month, m_day)

        sd_times = dtseries.apply(lambda dt: dt.replace(
                year=m_year, month=m_month, day=m_day))

        h_median.index = sd_times
        sd_list.append(h_median)

        # xs.append(list(h_median.index.astype(np.int64) // 10 ** 9))
        # ys.append(list(h_median))

        h_median.plot(ax=ax, c=colors[i], label=daylabel)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # ax.legend()  # pointless.. colorbar would be nicer.. try multiline?

    ax.set_title("Median Comb-Activity Mapped to the Same Day")

    # ffn = ioh.safename(path_out / f"{name}.png", "file")
    ffn = path_out / f"{name.lower()}_medians-sd"
    plot_path = ioh.safesavefig(ffn)
    logging.debug(f"Figure exported to {plot_path}")

    # # Try multiline
    # fig, ax = plt.subplots(figsize=resolution, dpi=100)
    # lc = multiline(xs, ys, colors,
    #                ax=ax, cmap="viridis", lw=1)
    # axcb = fig.colorbar(lc)
    # axcb.set_label("Date...")
    # ax.set_title("Median Comb-Activity Mapped to the Same Day")
    #
    # ffn = path_out / f"{name.lower()}_medians-sd-multiline"
    # plot_path = ioh.safesavefig(ffn)
    # logging.debug(f"Figure exported to {plot_path}")

    return plot_path


def fit_timeseries(xdates, ydata):
    """Fit sine function via fft or whatever.

    https://stackoverflow.com/questions/51637922/
    create-a-sine-wave-from-time-series-data-python

    https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/
    numpy.fft.fft.html#numpy.fft.fft

    https://stackoverflow.com/questions/55912403/
    predicting-sine-waves-in-python

    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.optimize.curve_fit.html
    """

    pass


def vec_dt_replace(series, year=None, month=None, day=None):
    """Use to cast all data to the same day.

    Do this in order to plot all data into the same 24-hour axis.

    From:
    https://stackoverflow.com/questions/28888730/pandas-change-day
    """
    return pd.to_datetime(
        {'year': series.dt.year if year is None else year,
         'month': series.dt.month if month is None else month,
         'day': series.dt.day if day is None else day})


def main(
    file_pattern=INFILE_PATTERN,
    # folder_pattern=INFOLDER_PATTERN,
    tol_td=TOLERANCE_TIMEDELTA,
    outlier=OUTLIER_THRESHOLD,
    args=ARGS,
):
    """Read image-difference CSVs into dataframes and make plots.

    Creates issues..

    INFO: Could not load matplotlib icon: can't use "pyimage10" as
          iconphoto: not a photo image
    See:
    https://github.com/matplotlib/matplotlib/issues/5963
    """
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

    # act_list = []
    # df_agg = None
    df_list = []
    med_list = []
    for csv_path in filelist:
        logging.info(f"Reading '{csv_path.name}'")

        hive, rpi, method, day_str = parse_filename(csv_path.name)
        name = f"RPi{rpi}_{day_str}_{method}"
        # Read CSV
        # header = [
        #     "time_central", "duration", "activity",
        #     "time1", "time2",
        #     "file1", "file2"
        # ]
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
        df["hour"] = df.index.hour
        df["hive"] = [hive] * len(df)
        df["rpi"] = [rpi] * len(df)
        df["method"] = [method] * len(df)

        # if df_agg is None:
        #     df_agg = df
        # else:
        #     df_agg = pd.concat([df_agg])

        # act_dict = {name: df["activity"]}
        #
        # act_list.append(act_dict)

        # Plot_single_activity day
        h_median = plot_single_activity(df["activity"], name, path_out)[1]

        # series = df.activity
        # series.index = series.index.hour
        hourly_bxpl_single(df, name, path_out)

        # Remove outliers
        if any(df.activity >= outlier):
            logging.warning(
                f"Found {sum(df.activity >= outlier)} outliers "
                f"in {csv_path.name}, filtering them out.")

            # Crop df to plausible measurements
            df = df[df.activity < outlier]

            name += "_removed-ols"

            # Plot_single_activity day
            h_median = plot_single_activity(
                    df["activity"], name, path_out)[1]

        df_list.append(df)
        med_list.append(h_median)

    df_agg = pd.concat(df_list)

    name = "aggregated"
    # name_euc = name + "_euclidean"
    # name_man = name + "_manhattan"

    # df_agg_euc = df_agg[df_agg.method == "euclidean"]
    # df_agg_man = df_agg[df_agg.method == "manhattan"]

    # Plot_single_activity day
    # plot_single_activity(df_agg_euc["activity"], name_euc, path_out)
    plot_single_activity(df_agg["activity"], name, path_out)

    # series = df.activity
    # series.index = series.index.hour

    # hourly_bxpl_single(df_agg_euc, name_euc, path_out)
    hourly_bxpl_single(df_agg, name, path_out)

    # Plot all medians
    plot_median_days(med_list, "median-days", path_out)

    # Plot functional median boxplot

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
