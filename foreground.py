########################################
# this file does not use pybgs library
########################################

########################################
########################################
# This sript will take video.MP4 exract
# a given area and calculate the pixel difference
# between each frame after n frames it
# takes the mean and plot it
# SETTINGS:
# you can jump frames forward to see variable jump_frames
# if you set save_plot the plots will get saved into plot folder
########################################
########################################

# Basic libraries
import numpy as np
import os
import glob
# open cv, we have to load V3 for full list of algorithms
# https://docs.opencv.org/3.4
import cv2 as cv
# graphic
import matplotlib.pyplot as plt

############## SETTINGS #######################
# plot style
plt.style.use('fast')
# our Video file
video_file = "video.MP4"
# create folder for plots and empty it
folder = "plot"
# jump frames forward
jump_frames = 7000
# save plot
save_plot = False
# threshold
threshold = 7
# mean interval for generating mean of pixel change and plotting steps, example 10 = each 10 frames make the mean and generate plot point
mean_interval = 10
# our rectangle area
x1 = 1130
x2 = 1800
y1 = 130
y2 = 900
# plot pixel color change break point
plot_breakpoint = 6
# adjust Gamma
adjustGamma = True
############## END SETTINGS #######################

# helper variables
w_pixel_array = []
ax = []
x_vec = []
y_vec = []
second_count = 0        # used to count frames
n_white_pix_sum = 0     # helper variable to sum white pixel in n amount of frames

print("###########  SETTINGS  ##################")

print("###########  PLOTS: {}".format(save_plot))
print("###########  PLOT breakpoint: {}".format(plot_breakpoint))
print("###########  Ignored frames: {}".format(jump_frames))
print("###########  Threshold: {}".format(threshold))
print("###########  Video: {}".format(video_file))
print("###########  X1: {}".format(x1))
print("###########  X2: {}".format(x2))
print("###########  Y1: {}".format(y1))
print("###########  Y2: {}".format(y2))
print("###########  Adjust Gamma: {}".format(adjustGamma))

# Check if video file exists
if(os.path.isfile(video_file) == False):
    print("Error: No video file found")
    exit()

# create output folder if not exists
try:
    os.mkdir(folder)
except:
    print("{} Folder already exists".format(folder))
    i = input("Clear files in {} folder (y/n)?".format(folder))
    if (i == "y"):
        # empty output folder
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print("Files deleted")

# load video
capture = cv.VideoCapture(video_file)
# wait till video is loaded
while not capture.isOpened():
    capture = cv.VideoCapture(video_file)
    cv.waitKey(1000)
    print("Wait for the header")

# jump forward in frames
capture.set(cv.CAP_PROP_POS_FRAMES, jump_frames)

length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
print( "Video total frames: {}".format(length) )

# https://makersportal.com/blog/2018/8/14/real-time-graphing-in-python
def live_plotter(x_vec, y1_data, line1, ax, frame, save, plot_breakpoint, mean_interval, pause_time = 0.04):

    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize = (8 , 4))
        ax = fig.add_subplot(1,1,1)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha = 0.8)
        #update plot label/title
        plt.ylabel('Moved pixels [%]')
        plt.xlabel('Frames [{} Frames Interval]'.format(mean_interval))
        plt.ylim(0,15,0.5)
        plt.show()

    plt.title('Pixel change {0:.2f}%'.format(y1_data[-1]))

    # change color if we drop lower than given threshold
    if(y1_data[-1] < plot_breakpoint):
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
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
       plt.ylim([np.min(0),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # if we want to save the plots
    if save:
        fname = 'plot/'+str(frame)+'.png'
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)    # return line so we can update it again in the next iteration
    return line1, ax

# build a lookup table mapping the pixel values [0, 255] to
# their adjusted gamma values
invGamma = 1.0 / 1.2
ltable = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")

# Somehow I found the value of `gamma=1.2` to be the best in my case
def adjust_gamma(image, ltable):
    # apply gamma correction using the lookup table
    return cv.LUT(image, ltable)

while True:
    # read video
    flag, frame = capture.read()

    # check if video is read
    if flag:
        # show video in ouput
        cv.namedWindow('original',cv.WINDOW_NORMAL)
        cv.resizeWindow('original', 600, 400)
        # draws a rectangle on frame

        # x1: y1, x2: y2
        cv.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv.imshow('original', frame)

        # crop only the area of interest for us
        # y1:y2, x1:x2
        roi = frame[y1:y2+1, x1:x2+1]
        cv.namedWindow('cropped',cv.WINDOW_NORMAL)
        cv.resizeWindow('cropped', 600, 400)
        cv.imshow('cropped', roi)

        pos_frame = capture.get(1)

        #### Preprocessing ######
        # change image to grayscale
        roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        if (adjustGamma == True):
            roi = adjust_gamma(roi, ltable)
        # equalize grayscale image
        #roi = cv.equalizeHist(roi)
        # add gaussian to remove noise
        #roi = cv.GaussianBlur(roi, (1, 1), 0)
        #roi = cv.medianBlur(roi, 1)
        #roi = cv.GaussianBlur(roi, (7, 7), 1.5)
        #### END Preprocessing ######

        cv.namedWindow('preprocess', cv.WINDOW_NORMAL)
        cv.resizeWindow('preprocess', 600, 400)
        cv.imshow('preprocess', roi)

        # check if it was the first run otherwise img_history is same as input for first round
        try:
            img_history
        except:
            img_history = roi

        # calculate absdiff
        img_output = cv.absdiff(roi, img_history)

        #### Output Processing ######
        #img_output = cv.cvtColor(img_output, cv.COLOR_BGR2GRAY)
        #img_output = cv.equalizeHist(img_output)
        #img_output = cv.GaussianBlur(img_output, (7, 7), 1.5)
        #img_output = cv.GaussianBlur(img_output, (3, 3), 1.5)
        img_output = cv.medianBlur(img_output, 1)


        # exports a black and white image
        _, img_output = cv.threshold(img_output, threshold, 255, cv.THRESH_BINARY)

        img_history = roi

        # show foreground mask in window
        cv.namedWindow('foreground', cv.WINDOW_NORMAL)
        cv.resizeWindow('foreground', 600, 400)
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
        if (second_count == mean_interval):
            # mean and relative value to all pixels of the cropped frame
            relative_white = (n_white_pix_sum / second_count) /  n_all_px * 100

            # add value our vector
            y_vec.extend([relative_white])
            x_vec.extend([pos_frame])

            # create our live plot
            w_pixel_array, ax = live_plotter(x_vec, y_vec, w_pixel_array, ax, pos_frame, save_plot, plot_breakpoint, mean_interval)

            # move our vector forward
            if (len(x_vec) > 250):
                y_vec.pop(0)
                x_vec.pop(0)

            # reset helper
            n_white_pix_sum = 0
            second_count = 0
            median_vec = []

            print('Number of mean white pixels: {0:.2f}%'.format(relative_white))

    else:
        #print "Frame is not ready"
        cv.waitKey(1000)
        break

    if 0xFF & cv.waitKey(10) == 27:
        break

# END
