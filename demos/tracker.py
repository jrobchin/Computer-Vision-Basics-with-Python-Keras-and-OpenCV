import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
import uuid # for generating unique file names
import math # math functions
import argparse

from IPython.display import display as ipydisplay, Image, clear_output, HTML # for interacting with the notebook better

import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

from utils import setup_tracker

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('src', default='video')
args = parser.parse_args()

try:
    src = int(args.src)
    video = cv2.VideoCapture(src)
except ValueError:
    if args.src == 'running':
        video = cv2.VideoCapture(os.path.join('images', 'running.mp4'))
    elif args.src == 'bottle':
        video = cv2.VideoCapture(os.path.join('images', 'moving_subject.mp4'))
    else:
        raise Exception('video not found')


cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

# Read first frame
success, frame = video.read()
if not success:
    print("first frame not read")
    sys.exit()

tracker = setup_tracker(4)

# Select roi for bbox
bbox = cv2.selectROI('frame', frame, False)
cv2.destroyAllWindows()

# Initialize tracker with first frame and bounding box
tracking_success = tracker.init(frame, bbox)

while True:
    time.sleep(0.02)
    
    timer = cv2.getTickCount()
    
    # Read a new frame
    success, frame = video.read()
    if not success:
        # Frame not successfully read from video capture
        break
        
    # Update tracker
    tracking_success, bbox = tracker.update(frame)
    
    # Draw bounding box
    if tracking_success:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)        
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    
    # Display result
    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("frame", frame)
    
    k = cv2.waitKey(1) & 0xff
    if k == 27: break # ESC pressed
        
cv2.destroyAllWindows()
video.release()