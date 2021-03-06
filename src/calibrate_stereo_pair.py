"""
User-interactive stereo calibration capture and computation application
"""
import logging
logger = logging.getLogger(__name__)
logger.debug('Importing')
from datetime import datetime
import sys
import os
import cv2
import argparse
import numpy as np
from stereo_calibrator import StereoCalibrator
from cal_target_defs import calibrationTargets
logger.debug('Done')

def readStereoData(leftUrl, rightUrl, savePath, leftFormat = 'left_%d.png', rightFormat = 'right_%d.png'):
    """
    User-interactive function, presents a window for esc/s/c
    Returns a list of pairs of paths like: [('savePath/left0.png','savePath/right0.png'), ...]
        that lists off all the files saved, or None if the user asked to exit
    leftUrl, rightUrl : string, URL to each camera stream, including authentication, compatible with cv2.VideoCapture(url)
    savePath : directory in which to save data files
    leftFormat, rightFormat : strings, formats with exactly one %d, which will be populated with an index starting at 0
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
            leftFn = os.path.join(savePath, leftFormat%imgNum)
            rightFn = os.path.join(savePath, rightFormat%imgNum)
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
    # This is super verbose
    logging.getLogger('stereo_calibrator').setLevel(logging.DEBUG)
    
    parser = argparse.ArgumentParser(description='Capture and/or process stereo calibration data.')
    parser.add_argument("-c", "--capture", help="Perform user-interactive image capture prior to processing", action="store_true", default=False)
    parser.add_argument("-p", "--process", help="Process images to compute stereo calibration", action="store_true", default=False)
    parser.add_argument("-l", "--leftUrl", metavar='URL', help="Left camera URL pattern, first %%s is replaced with username, second with password", default='http://%s:%s@192.168.0.253/mjpg/video.mjpg')
    parser.add_argument("-r", "--rightUrl", metavar='URL', help="Right camera URL pattern, first %%s is replaced with username, second with password", default='http://%s:%s@192.168.0.252/mjpg/video.mjpg')
    parser.add_argument("-f", "--folderNameFormat", metavar='Folder', help="Folder name full of images.  Will run strftime on this string, so you can use strftime escape characters.", default='stereo_cal_data_%Y-%d-%m_%H-%M-%S')
    parser.add_argument("-lf", "--leftFilenameFormat", metavar='Filename', help="Pattern for left-camera filenames within the specified folder.  Must have one %%d in it, which will receive an integer counting from 0.", default='left_%d.png')
    parser.add_argument("-rf", "--rightFilenameFormat", metavar='Filename', help="Pattern for right-camera filenames within the specified folder.  Must have one %%d in it, which will receive an integer counting from 0.", default='right_%d.png')
    parser.add_argument("-u", "--username", help="Username with which to log into camera")
    parser.add_argument("-pw", "--password", help="Password with which to log into camera")
    parser.add_argument("-t", "--target", metavar="index", help="Calibration target index (omit for user-interactive)")
    args = parser.parse_args()

    if not args.capture and not args.process:
        print("Need at least one of --capture or --process.")
        parser.print_help()
        sys.exit(0)
    
    # Application settings
    # leftUrlPattern = 'http://%s:%s@192.168.0.253/mjpg/video.mjpg'
    # rightUrlPattern = 'http://%s:%s@192.168.0.252/mjpg/video.mjpg'
    # folderNameFormat = 'stereo_cal_data_%Y-%d-%m_%H-%M-%S'
    leftUrlPattern = args.leftUrl
    rightUrlPattern = args.rightUrl
    folderNameFormat = args.folderNameFormat
    
    dataDir = datetime.now().strftime(folderNameFormat)
    if args.process:
        if args.target is not None:
            try:
                chosenTargetIdx = int(args.target)
            except:
                print("Must provide an integer for calibration target index, instead got %s."%args.target)
                sys.exit(0)
            if chosenTargetIdx >= len(calibrationTargets):
                print("Invalid calibration target index %d; exiting."%chosenTargetIdx)
                sys.exit(0)
        else:
            print("This program can seek the following targets: ")
            for i,target in enumerate(calibrationTargets):
                print('%2d: (%2dx%2d) "%s"s'%(i, target['dims'][0],  target['dims'][1], target['desc']))
            chosenTargetIdx = input("Enter number of target to use: ")
            try:
                chosenTargetIdx = int(chosenTargetIdx)
            except:
                pass
            if type(chosenTargetIdx) is not int or chosenTargetIdx >= len(calibrationTargets):
                print("Invalid ca selection; exiting.")
                sys.exit(0)
        calTarget = calibrationTargets[chosenTargetIdx]
    if args.capture:
        print("Will connect to (where %s:%s is the username and password):")
        print("Left cam: %s"%leftUrlPattern)
        print("Right cam: %s"%rightUrlPattern)
        print("Will save data to this directory: %s"%dataDir)
        print("During video stream, press the 's' key when the image is clear, the scene is stationary, and the target is visible in both images.")
        print("Press escape to exit.")
        if args.process:
            print("Press 'c' to process the captured images.")

    
        leftUrl = leftUrlPattern%(args.username,args.password)
        rightUrl = rightUrlPattern%(args.username,args.password)
        paths = readStereoData(leftUrl, rightUrl, dataDir, args.leftFilenameFormat, args.rightFilenameFormat)
    else:
        # Look through the indicated directory to find files
        allFilesInDir = [f for f in os.listdir(dataDir) if os.path.isfile(os.path.join(dataDir, f))]
        # print("allFilesInDir: " + repr(allFilesInDir))
        # We expect a pretty strict filename format here. 
        # Stop looping as soon as we run out of files.
        keepLooking = True
        i = 0
        paths = []
        while keepLooking:
            leftPath = os.path.join(dataDir, args.leftFilenameFormat%i)
            rightPath = os.path.join(dataDir, args.rightFilenameFormat%i)
            if os.path.isfile(leftPath) and os.path.isfile(rightPath):
                paths += [(leftPath, rightPath)]
                i += 1
            else:
                keepLooking = False
        print("Found %d stereo pairs in %s."%(i, dataDir))

    if args.process:
        if paths is not None and len(paths) > 0:
            print("Got paths: " + repr(paths))
            sc = StereoCalibrator(calTarget['dims'], calTarget['dotSpacingMm'], calTarget['simpleBlobDet'])
            
            stereo_cal = sc.find_stereo_pair_calibration(paths)
            if stereo_cal is not None:
                print("Have cal: " + repr(stereo_cal))
            else:
                print("Failed to find cal.")
        else:
            print("Commanded to process a calibration, but have no images to process.")
    print("Exiting.")
