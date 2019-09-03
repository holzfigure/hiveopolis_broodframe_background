########################################
########################################
# This sript will take video.MP4 exract
# a given area and calculate the pixel difference
# between each frame after n frames it
# takes the mean and plot it
# settings for pixel difference in
# /config/FrameDifference.xml
########################################
########################################

# Basic libraries
import numpy as np
import os
import glob

# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv
# our algorithm library please refer and citate https://github.com/andrewssobral/bgslibrary
# c++ but with python wrapper, has multiple algorithms which are not standard included in open cv
# does background image generation with the foremask from cv2
import pybgs as bgs
# timer are used for stats, too compare running time of algorithms
import time

# graphic
import matplotlib.pyplot as plt
plt.style.use('fast')

# https://makersportal.com/blog/2018/8/14/real-time-graphing-in-python
def live_plotter(x_vec, y1_data, line1, ax, frame, pause_time = 0.04):

    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize = (6 , 6))
        ax = fig.add_subplot(1,1,1)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha = 0.8)
        #update plot label/title
        plt.ylabel('Moved pixels [%]')
        plt.xlabel('Frames [5s Interval]')
        plt.title('Pixel change')
        plt.ylim(0,15,0.5)
        plt.show()

    # change color if we drop lower than given threshold
    if(y1_data[-1] < 2):
        ax.set_facecolor('xkcd:salmon')
    else:
        ax.set_facecolor('white')

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    line1.set_xdata(x_vec)

    plt.xlim(min(x_vec),max(x_vec), 5)

    plt.tick_params(
        axis = 'x',          # changes apply to the x-axis
        which = 'both',      # both major and minor ticks are affected
        bottom = True,      # ticks along the bottom edge are off
        top = False,         # ticks along the top edge are off
        labelbottom = True) # labels along the bottom edge are off

    # adjust limits if new data goes beyond bounds
    #if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
    #   plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    fname = 'plot/'+str(frame)+'.png'
    plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)    # return line so we can update it again in the next iteration
    return line1, ax

# Our used algorithm
algorithm = bgs.FrameDifference()
# our Video file
video_file = "video.MP4"

# helper arrays
w_pixel_array = []
ax = []
x_vec = []
y_vec = []

second_count = 0        # used to count frames
n_white_pix_sum = 0     # helper variable to sum white pixel in n amount of frames

# create folder for plots and empty it
folder = "plot"
try:
    os.mkdir(folder)
except:
    print("Folder is already created")

for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


# load video
capture = cv.VideoCapture(video_file)
# wait till video is loaded
while not capture.isOpened():
    capture = cv.VideoCapture(video_file)
    cv.waitKey(1000)
    print("Wait for the header")

while True:
    # read video
    flag, frame = capture.read()

    # check if video is read
    if flag:
        # show video in ouput
        cv.namedWindow('original',cv.WINDOW_NORMAL)
        cv.resizeWindow('original', 600, 600)
        # draws a rectangle on frame
        #cv.rectangle(frame, (700, 200), (1730, 750), (255,0,0), 2)
        cv.imshow('original', frame)

        # crop only the area of interest for us
        roi = frame[200:750+1, 700:1730+1]
        cv.namedWindow('cropped',cv.WINDOW_NORMAL)
        cv.resizeWindow('cropped', 600, 600)
        cv.imshow('cropped', roi)


        # TODO change X-Axis labels to frame number?
        pos_frame = capture.get(1)
        # apply given algroithm to get foregorund mask
        img_output = algorithm.apply(roi)
        # show foreground mask in window
        cv.namedWindow('foreground',cv.WINDOW_NORMAL)
        cv.resizeWindow('foreground', 600, 600)
        cv.imshow('foreground', img_output)

        # all pixels of image
        # TODO actually we would need this only one time, as all the frames have the same pixel count
        n_all_px = img_output.size
        # get all white pixels == changed pixels
        n_white_pix = np.sum(img_output == 255)
        # save into our helper variable
        n_white_pix_sum = n_white_pix_sum + n_white_pix
        # set our frame counter forward
        second_count = second_count + 1

        # if 10 frames we output the plot
        if (second_count == 5):
            # mean and relative value to all pixels of the cropped frame
            relative_white = (n_white_pix_sum / second_count) /  n_all_px * 100
            # add value our vector
            y_vec.append(relative_white)
            x_vec.append(pos_frame)

            # create our live plot
            w_pixel_array, ax = live_plotter(x_vec, y_vec, w_pixel_array, ax, pos_frame)

            # move our vector forward
            if (pos_frame > 500):
                y_vec.pop(0)
                x_vec.pop(0)

            # reset helper
            n_white_pix_sum = 0
            second_count = 0

            print('Number of mean white pixels:', relative_white)

    else:
        #print "Frame is not ready"
        cv.waitKey(1000)
        break

    if 0xFF & cv.waitKey(10) == 27:
        break

# END
