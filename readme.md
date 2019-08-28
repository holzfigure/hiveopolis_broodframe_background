## Algorithm testing to remove bees from timelapse of broodframe images ##

### Libraries ###
* python: glob; numby; openvc-python; pybgs

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

### video.py ###

Creates a video file from images in given folder

`python video.py FOLDERNAME`

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


