"""
User-interactive stereo calibration capture and computation application
"""
import logging
logger = logging.getLogger(__name__)
logger.info('Importing')
from datetime import datetime
import os
import cv2
import numpy as np
logger.info('Done')


def readVideoStream(url):
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame.")
            return
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) == 27:
            return

def readStereoData(leftUrl, rightUrl, savePath):
    """
    User-interactive function, presents a window for esc/s/c
    Returns a structure like: {'leftImagePaths':['file1',...], 'rightImagePaths':['file1',...]}
        that lists off all the files saved, or None if the user asked to exit
    leftUrl, rightUrl : string, URL to each camera stream, including authentication, compatible with cv2.VideoCapture(url)
    savePath : directory in which to save data files
    """
    os.makedirs(savePath)
    leftCap = cv2.VideoCapture(leftUrl)
    rightCap = cv2.VideoCapture(rightUrl)
    
    paths = {'leftImagePaths':[], 'rightImagePaths':[]}
    imgNum = 0
    while True:
        leftRet, leftFrame = leftCap.read()
        rightRet, rightFrame = rightCap.read()
        if not leftRet and rightRet:
            print("No frame.")
            return paths
            
        frame = np.hstack((leftFrame, rightFrame))
        cv2.imshow('Press escape to exit, s to save, c to calibrate based on saved', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            return None
        elif key == ord('c'):
            return paths
        elif key == ord('s'):
            pass # TODO
        else:
            # logger.debug('Got key: %d'%key)
        
            
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)

    from auth import USERNAME,PASSWORD # Make a file that defines these two strings
    # Application settings
    leftUrlPattern = 'http://%s:%s@192.168.0.253/mjpg/video.mjpg'
    rightUrlPattern = 'http://%s:%s@192.168.0.252/mjpg/video.mjpg'
    folderNameFormat = 'stereo_cal_data_%Y-%d-%m_%H-%M-%S'
    
    dataDir = datetime.now().strftime(folderNameFormat)
    
    print("Welcome to stereo pair calibration.  Will connect to (where %s:%s is the username and password):")
    print("Left cam: %s"%leftUrlPattern)
    print("Right cam: %s"%rightUrlPattern)
    print("Will save data to this directory: %s"%dataDir)
    print("During video stream, press the 's' key when the image is clear, the scene is stationary, and the target is visible in both images.")
    print("Press escape to exit.")
    print("Press 'c' to process the captured images.")

    
    leftUrl = leftUrlPattern%(USERNAME,PASSWORD)
    rightUrl = rightUrlPattern%(USERNAME,PASSWORD)
    readStereoData(leftUrl, rightUrl, dataDir)
    
