"""
User-interactive stereo calibration capture and computation application
"""
import logging
logger = logging.getLogger(__name__)
logger.debug('Importing')
from datetime import datetime
import os
import cv2
import numpy as np
from stereo_calibrator import StereoCalibrator
logger.debug('Done')


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
    Returns a list of pairs of paths like: [('savePath/left0.png','savePath/right0.png'), ...]
        that lists off all the files saved, or None if the user asked to exit
    leftUrl, rightUrl : string, URL to each camera stream, including authentication, compatible with cv2.VideoCapture(url)
    savePath : directory in which to save data files
    """
    leftCap = cv2.VideoCapture(leftUrl)
    rightCap = cv2.VideoCapture(rightUrl)
    
    paths = []
    imgNum = 0
    while True:
        leftRet, leftFrame = leftCap.read()
        # leftTimestamp = leftCap.get(cv2.CAP_PROP_POS_MSEC)
        rightRet, rightFrame = rightCap.read()
        # rightTimestamp = leftCap.get(cv2.CAP_PROP_POS_MSEC)
        if not leftRet and rightRet:
            print("No frame.")
            return paths
        
        # logger.info("Frame timestamps: %f, %f"%(leftTimestamp,rightTimestamp))
        
        frame = np.hstack((leftFrame, rightFrame))
        cv2.imshow('Press escape to exit, s to save, c to calibrate based on saved', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            return None
        elif key == ord('c'):
            return paths
        elif key == ord('s'):
            leftFn = os.path.join(savePath, 'left_%d.png'%imgNum)
            rightFn = os.path.join(savePath, 'right_%d.png'%imgNum)
            try:
                os.makedirs(savePath, exist_ok=True)
                cv2.imwrite(leftFn, leftFrame)
                cv2.imwrite(rightFn, rightFrame)
                paths += [(leftFn, rightFn)]
                logger.info("Saved pairs at %s,%s"%(leftFn, rightFn))
            except Exception as e:
                logger.error("Failed to save frames.", exc_info=True)
            imgNum += 1
            
        # else:
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
    paths = readStereoData(leftUrl, rightUrl, dataDir)
    if paths is not None:
        print("Got paths: " + repr(paths))
        sc = StereoCalibrator()
        
        stereo_cal = sc.find_stereo_pair_calibration(paths)
        if stereo_cal is not None:
            print("Have cal: " + repr(stereo_cal))
        else:
            print("Failed to find cal.")
    print("Exiting.")
