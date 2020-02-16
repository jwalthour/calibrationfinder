#!/usr/bin/python3
import logging
logger = logging.getLogger(__name__)
logger.debug('Importing')
import cv2
import numpy as np
logger.debug('Done')

class MonoCalibrator:
    def __init__(self, calPatternDims=(8, 8), calDotSpacingMm=(25.877, 25.877), detector=None):
        """
        calPatternDims : (int, int) - the rows,cols indicating the number of dots on the target
        calDotSpacingMm : (float,float) - in x,y, the number of millimeters between dots in x and y
        detector : a cv2.SimpleBlobDetector, or None for the default
        """
        if detector is None:
            self._calTargetDotDetector = self.makeDetector()
        else:
            self._calTargetDotDetector = detector
        
        # Set up the calibration pattern 
        self._calPatternDims = calPatternDims
        # self._calPatternDims = (24, 48)  # in dots, row,col
        self._calDotSpacingMm = calDotSpacingMm
        # self._calDotSpacingMm = (25.4, 25.4)  # in mm, x,y
        self._IMAGE_SIZE = (800,600)  # in px, x,y
        self._SENSOR_DIMS = (4*0.707107,4*0.707107)  # in mm, row,col
        self._cal3spacePattern = [] #[(x,y), ...]
        # OpenCV coordinate convention: x+ rightward, y+ downward, z+ out away from camera.
        for y in range(0, self._calPatternDims[0]):
            for x in range(0, self._calPatternDims[1]):
                self._cal3spacePattern += [(x * self._calDotSpacingMm[0], y * self._calDotSpacingMm[1], 0)]
        # logger.debug("self._cal3spacePattern: " + repr(self._cal3spacePattern))
    
    def makeDetector(self):
        # Setup SimpleBlobDetector parameters.
        parms = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        parms.minThreshold = 0;
        parms.maxThreshold = 128;
         
        # Filter by Area.
        parms.filterByArea = True
        parms.minArea = 5
         
        # Filter by Circularity
        parms.filterByCircularity = True
        parms.minCircularity = 0.25
         
        # Filter by Convexity
        parms.filterByConvexity = False
        parms.minConvexity = 0.9
        parms.maxConvexity = 1
         
        # Filter by Inertia
        parms.filterByInertia = True
        parms.minInertiaRatio = 0.5
        
        # logger.debug("Orig minDistBetweenBlobs: " + str(parms.minDistBetweenBlobs))
        parms.minDistBetweenBlobs = 5
        parms.blobColor = 0
         
        # Create a detector with the parameters
        return cv2.SimpleBlobDetector_create(parms)

    def findCameraCalibration(self, image_paths):
        """
        image_paths : list of image file paths
        return : cameraMatrix,distCoeffs if successful, or None,None if not
        """
        allPointsIn3space, allPointsInImages = self._findPointVectors(image_paths)
        if len(allPointsIn3space) > 0:
            # logger.debug("np.array(allPointsIn3space) = " + repr(np.array(allPointsIn3space)))
            allPointsIn3space = np.array(allPointsIn3space, dtype=np.float32)
            # logger.debug("allPointsIn3space = " + str(allPointsIn3space))
            # logger.debug("allPointsInImages = " + str(allPointsInImages))
            found,cameraMatrix,distCoeffs,rvecs,tvecs = cv2.calibrateCamera(allPointsIn3space, allPointsInImages, self._IMAGE_SIZE, None, None)
            
            # Debug by projecting the points in the calibration pattern onto the image
            # for img_path,points,rvec,tvec in zip(image_paths, allPointsIn3space, rvecs, tvecs):
                # img = cv2.imread(img_path)
                # imagePoints, jacobian = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)
                # self.drawPointsOnImage(img, imagePoints)
                # cv2.imshow('reprojected on %s'%img_path, img)
                # cv2.waitKey()
            
            # logger.debug("found: " + repr(found) + ",\n cameraMatrix: " + repr(cameraMatrix) + ",\n distCoeffs: " + repr(distCoeffs) + ",\n rvecs: " + repr(rvecs) + ",\n tvecs: " + repr(tvecs))
            # logger.debug("found: " + repr(found) + ",\n rvecs: " + repr(rvecs) + ",\n tvecs: " + repr(tvecs))
            return cameraMatrix,distCoeffs
        else: 
            logger.error("Can't find any calibration patterns in any of the supplied images.  Can't compute camera calibration.")
            return None,None
    
    def drawPointsOnImage(self, image, points, radius = 1, color = (0,0,255), drawNums = False): 
        """
        Annotate an image with points for debugging
        image : color opencv image
        points : list of coordinates in image
        radius: int, optional, marker circle size
        color: (b,g,r), opencv color
        drawNums: bool, optional, True to number the points
        """
        
        i = 0
        for point in points:
            # point is x,y, like : np.array([[697.77185, 396.0037 ]], dtype=float32
            # logger.debug("point: %s"%repr(point))
            cv2.circle(image, tuple(point[0]), radius, color, -1)
            cv2.putText(image, '%d'%i, tuple(point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.33, color)
            i += 1
    
    def _findPointVectors(self, image_paths, rowCol=False):
        """
        Get the coorinates of the dots on the calibration target
        
        image_paths : list of N image file paths
        rowCol : true to return points in row,col convention.  False to use x,y convention.
        returns : (<list of N copies of self._cal3spacePattern>, <list of arrays of dot coordinates in images>)
        """
        allPointsInImages = []
        allPointsIn3space = []
        
        first_loop = True
        for image_path in image_paths:
            img = cv2.imread(image_path)
            points = np.array([[]])
            found,points = cv2.findCirclesGrid(img, self._calPatternDims, points, cv2.CALIB_CB_SYMMETRIC_GRID, self._calTargetDotDetector)
            if found:
                # logger.debug("points.shape: %s"%repr(points.shape))
                points = points[:,0,:] # This doesn't seem to actually change anything, it seems to be just a spare dimension?
                # findCirclesGrid returns x,y convention.  Convert to row,col
                if rowCol:
                    points = points[:,[1,0]]
                # logger.debug("points.shape: %s"%repr(points.shape))
                # logger.debug(("Found " + str(len(points)) + " cal points in " + image_path) if found else "No cal pattern found in " + image_path)
                allPointsInImages += [points]
                allPointsIn3space += [self._cal3spacePattern]
                
                # self.drawPointsOnImage(img, points)
                # cv2.imshow(image_path, img)
            else:
                logger.warning("Didn't find calibration pattern in this image: %s"%image_path)
        # cv2.waitKey()
        
        return allPointsIn3space, allPointsInImages
    
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)

    mc = MonoCalibrator()
    cal_img_dir = 'test images/2019-10-18 stereo cal images/'
    left_image_names = [
        'left/left-00001.png',
        'left/left-00002.png',
        'left/left-00004.png',
        'left/left-00005.png',
        'left/left-00006.png',
        'left/left-00010.png',
        'left/left-00011.png',
        'left/left-00012.png',
        'left/left-00013.png',
        'left/left-00019.png',
        'left/left-00021.png',
        'left/left-00022.png',
        'left/left-00023.png',
    ]
    right_image_names = [
        'right/right-00001.png',
        'right/right-00002.png',
        'right/right-00004.png',
        'right/right-00005.png',
        'right/right-00006.png',
        'right/right-00010.png',
        'right/right-00011.png',
        'right/right-00012.png',
        'right/right-00013.png',
        'right/right-00019.png',
        'right/right-00021.png',
        'right/right-00022.png',
        'right/right-00023.png',
    ]
    left_paths  = [cal_img_dir + name for name in left_image_names]
    right_paths = [cal_img_dir + name for name in right_image_names]
    
    lCameraMatrix, lDistCoeffs = mc.findCameraCalibration(left_paths)
    rCameraMatrix, rDistCoeffs = mc.findCameraCalibration(right_paths)
    
    
    # Test/debug
    if lDistCoeffs is not None:
        logger.info("Got left calibration.")
        #undistort
        
    else:
        # Debug
        logger.info("Failed to calibrate left camera.")
    if rDistCoeffs is not None:
        logger.info("Got right calibration.")
        #undistort
        
    else:
        # Debug
        logger.info("Failed to calibrate right camera.")

    # cv2.waitKey()
