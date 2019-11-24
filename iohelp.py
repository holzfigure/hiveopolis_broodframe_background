#!/usr/bin/env python3
"""A library of helpful functions.

Notably to set up output-folders safely, with time-stamped copies
of the source code included.

holzfigure 2019
"""
# import os
# import csv
import time
import math
import shutil
# import argparse
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime  # , timedelta

import tkinter
from tkinter import Tk, filedialog
# import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# NOTE: module "imp" is deprecated..
import warnings
warnings.filterwarnings('error', category=DeprecationWarning)

# # GLOBALS
POSTFIX_DIR = "out"
TIME_FMT = "%y%m%d-%H%M%S-utc"
DAY_FMT = "%y%m%d-utc"

# Logging
LOG_MAX_BYTES = 20000000  # ~ 20 MB
LOG_BACKUP_COUNT = 50

# Plotting
DEF_EXT = "png"
DEF_WF = 3.0

COLORMAP = plt.cm.viridis
# colormap = plt.cm.viridis
# colormap = plt.cm.jet
# colormap = plt.cm.gist_ncar
# colormap = plt.cm.Set1


def now_str(pattern=TIME_FMT):
    """Return a formatted timestring for the current time."""
    # return time.strftime(pattern, time.gmtime())
    return datetime.utcnow().strftime(pattern)


def parse_subtree(filedir, pattern):
    """Parse a subtree (including subfolders) for the pattern.

    from:
    https://stackoverflow.com/questions/2186525/
    use-a-glob-to-find-files-recursively-in-python

    + sorting

    [requires 'import fnmatch']

    Deprecated since using pathlib! (v180817)
    """
    # matches = []
    # for root, dirnames, filenames in os.walk(filedir):
    #     for filename in fnmatch.filter(filenames, pattern):
    #         matches.append(os.path.join(root, filename))
    # return sorted(matches)
    filedir = Path(filedir).resolve()
    return sorted(filedir.rglob(pattern))


def safename(s, s_type="file"):
    """Append stuff to a file or folder if it already exists.

    Check whether a given file or folder 's' exists, return a non-existing
    filename.

    s ........ (full) filename or directory
    s_type ... 'file' or 'f' for files,
               'directory' or 'dir' or 'd' for folders

    Returns a file- or pathname that is supposedly safe to save
    without overwriting data.
    """
    # Ensure s is a Path object
    p = Path(s)

    low_type = str.lower(s_type)
    if low_type == "file" or low_type == "f":
        # if os.path.isfile(ss
        if p.is_file():
            stem = p.stem
            suffix = p.suffix
            counter = 0
            while p.is_file():
                # p = p.with_name(f"{stem}-{counter:02d}{suffix}")
                p = p.with_name("{}-{:02d}{}".format(stem, counter, suffix))
                counter += 1

    elif low_type == "directory" or low_type == "dir" or low_type == "d":
        if p.is_dir():
            stem = p.stem
            counter = 0
            while p.is_dir():
                # s = s_base + "-{:02d}".format(counter)
                # p = p.with_name(f"{stem}-{counter:02d}")
                p = p.with_name("{}-{:02d}".format(stem, counter))
                counter += 1
    return p


def safesavefig(path, ext=".png", close=True, verbose=False):
    """Safely save a figure from pyplot.

    adapted from:
    http://www.jesshamrick.com/2012/09/03/saving-figures-from-pyplot/

    # plt.gcf().canvas.get_supported_filetypes()
    # plt.gcf().canvas.get_supported_filetypes_grouped()
    filetypes = {
        'ps': 'Postscript',
        'eps': 'Encapsulated Postscript',
        'pdf': 'Portable Document Format',
        'pgf': 'PGF code for LaTeX',
        'png': 'Portable Network Graphics',
        'raw': 'Raw RGBA bitmap',
        'rgba': 'Raw RGBA bitmap',
        'svg': 'Scalable Vector Graphics',
        'svgz': 'Scalable Vector Graphics',
        'jpg': 'Joint Photographic Experts Group',
        'jpeg': 'Joint Photographic Experts Group',
        'tif': 'Tagged Image File Format',
        'tiff': 'Tagged Image File Format'
    }

    180817  Added a '.' to the default extension to be compatible
            with path.suffix
    """
    valid_extensions = plt.gcf().canvas.get_supported_filetypes()
    fallback_ext = ".png"

    # Ensure path is a pathlib.Path object
    path = Path(path)

    # Parse path components
    directory = path.parent
    stem = path.stem
    suffix = path.suffix

    # Check whether path already has an extension
    if suffix:
        if suffix in valid_extensions:
            if suffix != ext:
                logging.debug(f"Overwriting kwarg ext '{ext}' " +
                              f"with suffix '{suffix}' from {path}!")
            ext = suffix
        else:
            logging.debug(f"Overwriting file suffix '{suffix}' "
                          f"with kwarg ext '{ext}'!")

    # Ensure extension is correct
    ext = ext.lower()
    if not ext.startswith("."):
        logging.debug(f"Adding '.' to {ext}")
        ext = f".{ext}"
    if ext.split(".")[-1] not in valid_extensions:
        logging.warning(f"Invalid extension '{ext}', " +
                        f"replacing with '{fallback_ext}'")
        ext = fallback_ext

    # Generate filename
    # filename = "%s.%s" % (os.path.split(path)[1], ext)
    filename = stem + ext

    # Ensure valid directory
    if not directory:
        directory = Path.cwd()
    directory = directory.resolve()
    if not directory.is_dir():
        directory.mkdir(parents=True)

    # Finalize full filename
    # savepath = os.path.join(directory, filename)
    savepath = directory / filename
    savepath = safename(savepath, 'file')

    # Save figure to file
    # TODO: Remove str() once matplotlib is updated??
    plt.savefig(str(savepath))
    if verbose:
        logging.info(f"Saved figure to {savepath}")

    if close:
        plt.close()
    # if verbose:
    #     logging.debug("Done")
    return savepath


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.

    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}

    Code adapted from
    http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    Width and max height in inches for IEEE journals taken from
    https://www.computer.org/cms/Computer.org/Journal%20templates/
    transactions_art_guide.pdf

    from https://nipunbatra.github.io/blog/2014/latexify.html
    (v180817: updated this link)

    """
    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (math.sqrt(5) - 1.0) / 2.0  # aesthetic ratio
        fig_height = fig_width * golden_mean   # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 6,   # fontsize for x and y labels (was 10)
              'axes.titlesize': 6,
              'font.size': 6,  # 'text.fontsize': 8,    # was 10
              'legend.fontsize': 6,  # was 10
              'xtick.labelsize': 6,
              'ytick.labelsize': 6,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
              }

    matplotlib.rcParams.update(params)


def format_axes(ax):
    """Format axes."""
    spine_color = 'gray'

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=spine_color)

    return ax

    # # The following functions require numpy:
    #    def euclid(p1, p2):
    #        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    #
    #    def euclid_array(p1s, p2s):
    #        """The inputs "p1s" and "p2s" are 2-column arrays
    #        of XY-coordinates.
    #        """
    #        return np.sqrt((p2s[:, 0] - p1s[:, 0])**2 +
    #                       (p2s[:, 1] - p1s[:, 1])**2)


def setup_environment(
        thisfile,
        dir_targ=None,
        level=1,
        new_dir=True,
        prefix_file=None,
        postfix_dir=None,
        daystamp=False, day_fmt=DAY_FMT,
        dependencies=None,
        ):
    """Create an output directory safely.

    No overwriting of existing files and folders.

    in (optional):

        dir_targ    ....    str     full path to a directory. def=""
        level       ....    int     (-1, [0], 1) are the choices.
                        [-1: dir-out is sibling to the given directory
                          0: dir_out is the given directory
                             CAUTION: will only be this directory if
                                      new_dir=False and postfix=None!
             DEFAULT:     1: dir_out is child of the given directory]

        new_dir     ....    bool    if True, create a new directory, even if
                                      one already exists
                                    if False, write into an existing
                                      directory with the given name
        thisfile    ....    bool    if True, get full path to current file
                        [if i call "os.path.basename(__file__)" here,
                         will i get the path to the calling code,
                         or to this file 'holzhelp.py'?]
        prefix_file  ....    str     prefix file with this (and '_')
        postfix_dir  ....    str     append to name of output-folder.
        dependencies ...    list    paths to other files to copy to dir_out

    out:
        dir_out     ....    str     full path to created output directory
    """
    # Set up directories
    # ==================

    # # if interactive:
    # dir_targ = filedialog.askdirectory(initialdir=DIR_INI)

    # thisfile = os.path.basename(__file__)
    # thisname = os.path.splitext(os.path.split(thisfile)[1])[0]
    thisfile = Path(thisfile).resolve()
    thisname = thisfile.stem
    if prefix_file:
        # thisname = f"{prefix_file}_{thisname}"
        thisname = "{}_{}".format(prefix_file, thisname)

    if not dir_targ:
        # dir_targ = os.path.join(os.getcwd(), postfix)
        # dir_targ = os.getcwd()
        # dir_targ = Path.cwd() / f"{thisname}_{postfix_dir}"
        dir_targ = Path.cwd()
    else:
        dir_targ = Path(dir_targ)

    # determine level to place directory
    if level < 0:
        # basedir, lastdir = os.path.split(dir_targ)
        # os.path.join(basedir, thisname)
        # dir_out = dir_targ.with_name(f"{dir_targ.stem}_{thisname}")
        dir_out = dir_targ.with_name("{}_{}".format(
            dir_targ.stem, thisname))
    elif level == 0:
        # NOTE: only stays if new_dir=False and postfix=None!
        dir_out = dir_targ
    elif level > 0:
        # dir_out = os.path.join(dir_targ, thisname)
        dir_out = dir_targ / thisname

    if postfix_dir:
        # dir_out += "_" + postfix_dir
        # dir_out = dir_out.with_name(f"{dir_out.stem}_{postfix_dir}")
        dir_out = dir_out.with_name("{}_{}".format(
            dir_out.stem, postfix_dir))
    if daystamp:
        # dir_out += now_str("_%y%m%d-utc")
        # dir_out = dir_out.with_name(f"{dir_out.stem}_{now_str(day_fmt)}")
        dir_out = dir_out.with_name("{}_{}".format(
            dir_out.stem, now_str(day_fmt)))
    if new_dir:
        dir_out = safename(dir_out, 'directory')

    if not dir_out.is_dir():
        # os.mkdir(dir_out)
        dir_out.mkdir(parents=True)
        # logging.info("created output directory at '{}'".format(dir_out))
        # logwarn = []
    # else:
    #     logwarn = ("output directoy already exists, " +
    #                "error in function safename()")

    # copy files to output-directory
    src_out = dir_out / "src"
    if not src_out.is_dir():
        src_out.mkdir()
        print(f"Created folder '{src_out}'")
    if not dependencies:
        dependencies = []
    dependencies.append(thisfile)
    for filename in dependencies:
        # path, fname = os.path.split(filename)
        # name, ext = os.path.splitext(fname)
        # path = filename.parent
        filename = Path(filename).resolve()
        name = filename.stem
        suffix = filename.suffix
        if prefix_file:
            # name = f"{prefix_file}_{name}"
            name = "{}_{}".format(prefix_file, name)
        # thatfile = os.path.join(
        #     dir_out, name + now_str() + ext)
        # thatfile = dir_out / f"{name}_{now_str()}{suffix}"
        thatfile = src_out / "{}_{}{}".format(name, now_str(), suffix)
        thatfile = safename(thatfile, 'file')
        # TODO: Replace this with a proper pathlib method once?
        #       And remove the 'str()' once Raspbian is n Python 3.6..
        shutil.copy2(str(filename), str(thatfile))

    #    this_split = os.path.splitext(thisfile)
    #    thatfile = os.path.join(
    #        dir_out, this_split[0] + now_str() + this_split[1])
    #    thatfile = safename(thatfile, 'file')
    #    shutil.copy2(thisfile, thatfile)

    return dir_out, thisname  # , logwarn


def setup_logging(
        thisname,
        args,
        dir_log=None,
        max_bytes=LOG_MAX_BYTES,
        backup_count=LOG_BACKUP_COUNT,
        ):
    """Set up the logging module to log to a file.

    Rotate logfiles if they are bigger than LOG_MAX_BYTES.

    https://docs.python.org/3/howto/logging-cookbook.html
    """
    err_msg = []
    if dir_log is None:
        # dir_log = os.path.join(os.getcwd(), "DIR_LOG")
        dir_log = Path.cwd() / "log"
        dir_log = safename(dir_log, 'dir')
    if not dir_log.is_dir():
        try:
            dir_log.mkdir(parents=False)
        except Exception as err:
            # err_msg.append(
            #     f"Failed to create directory {dir_log}\n" +
            #     f"Error: {err}\n" +
            #     "Now creating full path...")
            err_msg.append((
                "Failed to create directory {}\n" +
                "Error: {}\n" +
                "Now creating full path...").format(dir_log, err))
            dir_log.mkdir(parents=True)
    # log_path = os.path.join(LOC_PATH, "logs")

    # thisfile = os.path.basename(__file__)
    # logfile = safename(os.path.join(
    #     dir_log, "{}_{}.log".format(thisname, now_str())), 'file')
    logfile = safename(
        # (dir_log / f"{thisname}_{now_str()}.log"), 'file')
        (dir_log / "{}_{}.log".format(thisname, now_str())), 'file')
    # logfile = safename(logfile, 'file')
    # logname = thisfile[0:-3] + '.log'  # + now_str() + '.log'
    if args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    #    logging.basicConfig(
    #            level=loglevel,
    #            format=("%(asctime)s %(levelname)-8s " +
    #                    "%(funcName)-12s: %(message)s"),
    #            datefmt='%y-%m-%d %H:%M:%S UTC',
    #            filename=logfile,
    #            filemode='a')
    #    # logging.basicConfig(filename=logfile, level=logging.INFO)
    #    logging.debug("logging to file {}".format(logfile))

    # Set level
    logging.getLogger('').setLevel(loglevel)

    # All times in UTC
    logging.Formatter.converter = time.gmtime
    # format=('%(asctime)s %(name)-12s %(levelname)-8s %(message)s',

    # Rotating logs
    # https://docs.python.org/2/howto/
    #         logging-cookbook.html#using-file-rotation
    # Add the log message handler to the logger
    # TODO: Remove the "str()" once RPIs have Python3.6
    rotater = logging.handlers.RotatingFileHandler(
        str(logfile),
        mode='a',
        maxBytes=max_bytes,
        backupCount=backup_count)
    #     encoding=None,
    #     delay=0)
    # rotater.setLevel(loglevel)
    rotate_formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(funcName)-12s: %(message)s",
        datefmt='%y-%m-%d %H:%M:%S UTC')
    rotater.setFormatter(rotate_formatter)
    logging.getLogger('').addHandler(rotater)

    # if not cron:
    # Define a Handler which writes INFO messages or
    # higher to the sys.stderr
    console = logging.StreamHandler()
    # console.setLevel(loglevel)  # (logging.INFO)
    # Set a format which is simpler for console use
    # formatter = logging.Formatter(
    #         '%(name)-12s: %(levelname)-8s %(message)s')
    console_formatter = logging.Formatter(
        "%(levelname)-8s: %(message)s")
    # Tell the handler to use this format
    console.setFormatter(console_formatter)
    # Add the handler to the root logger
    logging.getLogger('').addHandler(console)

    if len(err_msg) > 0:
        for msg in err_msg:
            logging.warning(msg)
    logging.debug("Logging to screen and to {}".format(logfile))
    # return dir_log


def select_files(title="select file(s)", dir_ini=None,
                 filetypes=[("all files", "*")],
                 more=False):
    """Interactively pick a file (actually its path-string).

    If 'more=True', a tuple of files will be returned.

    see:
    http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/tkFileDialog.html

    http://www.programcreek.com/python/example/4281/
        tkFileDialog.askopenfilename

    http://effbot.org/tkinterbook/tkinter-file-dialogs.htm

    Not mentioned in the above refs is ".askopenfilenames()",
    which takes the same options but returns a tuple of
    selected files.

    >> dir(filedialog)
    ['ACTIVE', 'ALL', 'ANCHOR', 'ARC', 'BASELINE', 'BEVEL', 'BOTH',
    'BOTTOM', 'BROWSE', 'BUTT', 'BaseWidget', 'BitmapImage', 'BooleanVar',
    'Button', 'CASCADE', 'CENTER', 'CHAR', 'CHECKBUTTON', 'CHORD', 'COMMAND',
    'CURRENT', 'CallWrapper', 'Canvas', 'Checkbutton', 'DISABLED', 'DOTBOX',
    'Dialog', 'Directory', 'DoubleVar',
    'E', 'END', 'EW', 'EXCEPTION', 'EXTENDED', 'Entry', 'Event', 'EventType',
    'FALSE', 'FIRST', 'FLAT', 'FileDialog', 'Frame', 'GROOVE', 'Grid',
    'HIDDEN', 'HORIZONTAL', 'INSERT', 'INSIDE', 'Image', 'IntVar',
    'LAST', 'LEFT', 'Label', 'LabelFrame', 'Listbox', 'LoadFileDialog',
    'MITER', 'MOVETO', 'MULTIPLE', 'Menu', 'Menubutton', 'Message',
    'Misc', 'N', 'NE', 'NO', 'NONE', 'NORMAL', 'NS', 'NSEW, 'NUMERIC',
    'NW', 'NoDefaultRoot', 'OFF', 'ON', 'OUTSIDE', 'Open', 'OptionMenu',
    'PAGES', 'PIESLICE', 'PROJECTING', 'Pack', 'PanedWindow', 'PhotoImage',
    'Place', 'RADIOBUTTON', 'RAISED', 'READABLE', 'RIDGE', 'RIGHT', 'ROUND',
    'Radiobutton', 'S', 'SCROLL', 'SE', 'SEL', 'SEL_FIRST', 'SEL_LAST',
    'SEPARATOR', 'SINGLE', 'SOLID', 'SUNKEN', 'SW',
    'SaveAs', 'SaveFileDialog', 'Scale', 'Scrollbar', 'Spinbox', 'StringVar',
    'TOP', 'TRUE', 'Tcl', 'TclError', 'TclVersion', 'Text', 'Tk', 'TkVersion',
    'Toplevel', 'UNDERLINE', 'UNITS', 'VERTICAL', 'Variable', 'W', 'WORD',
    'WRITABLE', 'Widget', 'Wm', 'X', 'XView', 'Y', 'YES', 'YView', '_Dialog',
    '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
    '__name__', '__package__', '__spec__',
    'askdirectory', 'askopenfile', 'askopenfilename', 'askopenfilenames',
    'askopenfiles', 'asksaveasfile', 'asksaveasfilename', 'commondialog',
    'constants', 'dialogstates', 'enum', 'fnmatch',
    'getboolean', 'getdouble', 'getint', 'image_names', 'image_types',
    'mainloop', 'os', 're', 'sys', 'test', 'wantobjects']
    """
    # Resolve initial directory
    if not dir_ini:  # or not dir_ini.is_dir():
        dir_ini = Path.cwd()
    else:
        dir_ini = Path(dir_ini).resolve()

    # Include this to make the crappy empty window go away
    root = Tk()
    root.withdraw()

    print(f"tkinter version: {tkinter.TkVersion}")

    # Set options
    opts = {}
    opts["parent"] = root
    opts["title"] = title
    opts["initialdir"] = dir_ini
    opts['filetypes'] = filetypes

    # Check whether single file or tuple of files is requested
    if more:
        # tuple of full filenames (paths)
        # ffn_return = tkFileDialog.askopenfilenames(**opts)
        ffn_return = filedialog.askopenfilenames(**opts)
        if len(ffn_return) > 0:
            ffn_return = [Path(ffn) for ffn in ffn_return]

    else:
        # String of full filename (path)
        # ffn_return = tkFileDialog.askopenfilename(**opts)
        ffn_return = filedialog.askopenfilename(**opts)
        if ffn_return:
            ffn_return = Path(ffn_return)

    # If cancelled, return None
    if not ffn_return:
        return None

    # Return full filename(s)
    return ffn_return


def select_directory(title="select directory", dir_ini=None):
    """Interactively retrieve the path to a directory."""
    # include this to make the crappy empty window go away
    root = Tk()
    root.withdraw()

    print(f"tkinter version: {tkinter.TkVersion}")

    # open directory dialog
    # dir_select = tkFileDialog.askdirectory(
    dir_select = filedialog.askdirectory(
        parent=root,
        title=title,
        initialdir=dir_ini)

    # check cancel or false directoy
    if not dir_select:
        print("Cancelled by user, returning 'None'")
        return None
    else:
        dir_select = Path(dir_select)
    if not dir_select.is_dir():
        print(f"Directory '{dir_select}' doesn't exist, returning 'None'")
        return None

    # return full path of selected diretory
    return dir_select


def main():
    """Mock main-function.

    Write test cases here.
    """
    # setup environment
    # thisfile = os.path.basename(__file__)
    # dir_out = setup_environment(thisfile, postfix=postfix)
    pass


if __name__ == "__main__":
    main()
