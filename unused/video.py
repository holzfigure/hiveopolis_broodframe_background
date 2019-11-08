########################################
########################################
# File to create video from our created
# images in given folder
########################################
########################################

# Libraries
import numpy as np
import os
import cv2 as cv
import glob
import sys

print(sys.argv)
if len(sys.argv) == 0 or len(sys.argv) > 5:
    print("No image name given or to many, only one allowed")
    sys.exit()

if os.path.isdir(str(sys.argv[1])) != True:
    print("Given directory does not exist: " + str(sys.argv[1]))
    sys.exit()

img_folder = str(sys.argv[1]) + "/*.jpg"
video_file = str(sys.argv[2]) + ".avi"

# Create holding array
img_array = []
img_array = sorted(glob.iglob(img_folder), key = os.path.getctime, reverse = False)
img_out_array = []
#print(img_array)
print("Start creating video ... " + img_folder)

# Loop over all files in our bg folder
for x in range(0, len(img_array)):
    # read image and size for generating video
    # TODO size we need only once because all are the same size but doesnt matter i think
    img = cv.imread(img_array[x])
    #print(img)
    height, width, layers = img.shape
    size = (width, height)
    # add to array
    img_out_array.append(img)

print("All files inside our array, total length " + str(len(img_array)))


# create our video container
out = cv.VideoWriter(video_file, cv.VideoWriter_fourcc(*'DIVX'), 15, size)

print("Video container created")

counter = 0
for i in img_out_array:
    print("Write image to video " + str(counter))
    out.write(i)
    counter += 1

print("Finished, releasing file")
out.release()

# END
