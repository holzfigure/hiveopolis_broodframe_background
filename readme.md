## Algorithm testing to remove bees from timelapse of broodframe images ##

### Libraries ###
`glob; numby; openvc-python; pybgs`

We need to use opencv version 3.x for best support with bgslibrary, see issue: https://github.com/andrewssobral/bgslibrary/issues/164

To install and use pybgs manually:

`brew install opencv@3`

```
git clone --recursive https://github.com/andrewssobral/bgslibrary.git

cd bgslibrary
cd build

cmake -D BGS_PYTHON_SUPPORT=ON -D OpenCV_DIR="your installed OpenCV directory" ..
make install
```

### benchmark.py  ###

This python script runs multiple algorithms and creates the calculated backgroundimages and the writes the time to finish in a text file.
`python benchmark.py`

### foreground.py  ###

Live foreground mask and pixeldifference (absdifference) as plot from 'video.MP4' in main folder. It will save the plot as png into the plot folder. Each 5 frames are one generated plot images, this means for a 25fps Videos it will create 5 plot images.
`python foreground.py`

### foreground_pybgs.py  ###

Pybgs library is needed for this script. Here we can test various algorithms which are included in pybgs to check different solutions to frame differences. Simply change the algorithm in code. Config files are in config folder.
`python foreground_pybgs.py`

### gaussian.py  ###

Uses gaussian method to remove moving object from image frames. Settings can be set inside the script and some option to preprocess the images are given, but commented out. 
`python gaussian.py`

### video.py ###

Creates a video file from images in given folder

`python video.py FOLDERNAME FILENAME`

### references and citation ###

https://github.com/opencv/opencv
```
@article{opencv_library,
    author = {Bradski, G.},
    citeulike-article-id = {2236121},
    journal = {Dr. Dobb's Journal of Software Tools},
    keywords = {bibtex-import},
    posted-at = {2008-01-15 19:21:54},
    priority = {4},
    title = {{The OpenCV Library}},
    year = {2000}
}
```

https://github.com/andrewssobral/bgslibrary
```
@inproceedings{bgslibrary,
author    = {Sobral, Andrews},
title     = {{BGSLibrary}: An OpenCV C++ Background Subtraction Library},
booktitle = {IX Workshop de Visão Computacional (WVC'2013)},
address   = {Rio de Janeiro, Brazil},
year      = {2013},
month     = {Jun},
url       = {https://github.com/andrewssobral/bgslibrary}
}
```
