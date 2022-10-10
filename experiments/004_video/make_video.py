"""
Make video from png files
"""

import glob
import cv2
import os


# get list of files
folder = 'experiments/004_video/images'
images = glob.glob(folder + "/*_val.png")
images = sorted(images)


video_name = 'experiments/004_video/video.avi'



frame = cv2.imread(images[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video = cv2.VideoWriter(video_name, fourcc, 3, (width,height))

for image in images:
    print(image)
    video.write(cv2.imread(image))

cv2.destroyAllWindows()
video.release()