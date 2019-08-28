## Algorithm testing to remove bees from timelapse of broodframe images ##

### Libraries ###
* python: glob; numby; openvc-python; pybgs

To install and use pybgs with pip -- we need full c++ library of opencv version 3

`brew install opencv3`

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


