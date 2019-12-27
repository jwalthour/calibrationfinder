#!/usr/bin/python3
import logging
logger = logging.getLogger(__name__)
logger.info('Importing')
import cv2
import cv2.aruco
import numpy as np

class StereoCalibrator:
    def __init__(self):
        """
        """
        self._cal_target_dot_det = self.make_detector()
        
        # Set up the calibration pattern 
        self.CAL_PATTERN_DIMS = (8, 8)  # in dots, row,col
        self.CAL_DOT_SPACING_MM = (25.877, 25.877)  # in mm, x,y
        # self._IMAGE_SIZE = (600,800)  # in px, row,col
        self._IMAGE_SIZE = (800,600)  # in px, x,y
        self._SENSOR_DIMS = (4*0.707107,4*0.707107)  # in mm, row,col
        self._cal_3space_pattern = [] #[(x,y), ...]
        # OpenCV coordinate convention: x+ rightward, y+ downward, z+ out away from camera.
        for y in range(0, self.CAL_PATTERN_DIMS[0]):
            for x in range(0, self.CAL_PATTERN_DIMS[1]):
                self._cal_3space_pattern += [(x * self.CAL_DOT_SPACING_MM[0], y * self.CAL_DOT_SPACING_MM[1], 0)]
        # logger.debug("self._cal_3space_pattern: " + repr(self._cal_3space_pattern))
    
    def make_detector(self):
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

    def find_single_cam_calibration(self, image_paths):
        """
        image_paths : list of image file paths
        """
        all_points_in_3space, all_points_in_images = self._find_point_vectors(image_paths)
        if len(all_points_in_3space) > 0:
            # logger.debug("np.array(all_points_in_3space) = " + repr(np.array(all_points_in_3space)))
            all_points_in_3space = np.array(all_points_in_3space, dtype=np.float32)
            # logger.debug("all_points_in_3space = " + str(all_points_in_3space))
            # logger.debug("all_points_in_images = " + str(all_points_in_images))
            found,cameraMatrix,distCoeffs,rvecs,tvecs = cv2.calibrateCamera(all_points_in_3space, all_points_in_images, self._IMAGE_SIZE, np.array([[]]), np.array([]))
            
            # Debug by projecting the points in the calibration pattern onto the image
            # for img_path,points,rvec,tvec in zip(image_paths, all_points_in_3space, rvecs, tvecs):
                # img = cv2.imread(img_path)
                # imagePoints, jacobian = cv2.projectPoints(points, rvec, tvec, cameraMatrix, distCoeffs)
                # self._draw_points_on_image(img, imagePoints)
                # cv2.imshow('reprojected on %s'%img_path, img)
                # cv2.waitKey()
            
            # logger.debug("found: " + repr(found) + ",\n cameraMatrix: " + repr(cameraMatrix) + ",\n distCoeffs: " + repr(distCoeffs) + ",\n rvecs: " + repr(rvecs) + ",\n tvecs: " + repr(tvecs))
            # logger.debug("found: " + repr(found) + ",\n rvecs: " + repr(rvecs) + ",\n tvecs: " + repr(tvecs))
        else: 
            logger.error("Can't find any calibration patterns in any of the supplied images.  Can't compute single camera calibration.")
        return cameraMatrix,distCoeffs
    
    def _draw_points_on_image(self, image, points): 
        """
        Annotate an image with points for debugging
        image : color opencv image
        points : list of coordinates in image
        """
        RADIUS = 1
        COLOR = (0,0,255)
        i = 0
        for point in points:
            # point is x,y, like : np.array([[697.77185, 396.0037 ]], dtype=float32
            # logger.debug("point: %s"%repr(point))
            cv2.circle(image, tuple(point[0]), RADIUS, COLOR, -1)
            cv2.putText(image, '%d'%i, tuple(point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLOR)
            i += 1
    
    def _find_point_vectors(self, image_paths):
        """
        image_paths : list of N image file paths
        returns : (<list of N copies of self._cal_3space_pattern>, <list of lists of dot coordinates in images>)
        """
        all_points_in_images = []
        all_points_in_3space = []
        
        first_loop = True
        for image_path in image_paths:
            img = cv2.imread(image_path)
            points = np.array([[]])
            found,points = cv2.findCirclesGrid(img, self.CAL_PATTERN_DIMS, points, cv2.CALIB_CB_SYMMETRIC_GRID, self._cal_target_dot_det)
            # logger.debug(("Found " + str(len(points)) + " cal points in " + image_path) if found else "No cal pattern found in " + image_path)
            if found:
                all_points_in_images += [points]
                all_points_in_3space += [self._cal_3space_pattern]
                
                # self._draw_points_on_image(img, points)
                # cv2.imshow(image_path, img)
            else:
                logger.warning("Didn't find calibration pattern in this image: %s"%image_path)
        # cv2.waitKey()
        
        return all_points_in_3space, all_points_in_images
    
    def find_stereo_pair_calibration(self, left_image_paths, right_image_paths, pair_image_paths):
        """
        
        left_image_paths  : list of strings, each of which is a path to an image from the left camera
        right_image_paths : list of strings, each of which is a path to an image from the right camera
        pair_image_paths  : list of twoples, of the form ("/path/to/one/left/image", "/path/to/one/right/image"),
        
        returns : Projection matrices for cv2.triangulatePoints, stored in a dictionary like this:
        {
            'leftProjMat':leftProjMat ,
            'rightProjMat':rightProjMat,
        }
        """
        # First must calibrate individual cameras
        logger.info("Computing left camera calibration")
        lCameraMatrix, lDistCoeffs = self.find_single_cam_calibration(left_image_paths)
        logger.info("lCameraMatrix: " + repr(lCameraMatrix))
        logger.info("Computing right camera calibration")
        rCameraMatrix, rDistCoeffs = self.find_single_cam_calibration(right_image_paths)
        logger.info("rCameraMatrix: " + repr(rCameraMatrix))
        
        # redefine these 
        left_image_paths = [pair[0] for pair in pair_image_paths]
        right_image_paths = [pair[1] for pair in pair_image_paths]
        
        # Find individual dots in all the images
        logger.info("Finding points in left images from pairs")
        all_points_in_3space, all_points_in_left_images = self._find_point_vectors(left_image_paths)
        logger.info("Finding points in right images from pairs")
        all_points_in_3space, all_points_in_right_images = self._find_point_vectors(right_image_paths)
        all_points_in_3space = np.array(all_points_in_3space, dtype=np.float32)
        # logger.debug("all_points_in_3space: " + repr(all_points_in_3space))
        # logger.debug("all_points_in_left_images: " + repr(all_points_in_left_images))
        # logger.debug("all_points_in_right_images: " + repr(all_points_in_right_images))
        # logger.debug("self._IMAGE_SIZE: " + repr(self._IMAGE_SIZE))
        logger.debug("len(all_points_in_3space): " + str(len(all_points_in_3space)))
        logger.debug("len(all_points_in_left_images): " + str(len(all_points_in_left_images)))
        logger.debug("len(all_points_in_right_images): " + str(len(all_points_in_right_images)))
        logger.info("Computing stereo calibration")
        minError, lCameraMatrixUpdated, lDistCoeffsUpdated, rCameraMatrixUpdated, rDistCoeffsUpdated, R, T, E, F = cv2.stereoCalibrate(all_points_in_3space, all_points_in_left_images, all_points_in_right_images, lCameraMatrix, lDistCoeffs, rCameraMatrix, rDistCoeffs, self._IMAGE_SIZE, flags=cv2.CALIB_FIX_INTRINSIC)
        logger.debug("minError: " + repr(minError))
        logger.debug("lCameraMatrix: " + repr(lCameraMatrix) + " -> lCameraMatrixUpdated: " + repr(lCameraMatrixUpdated))
        logger.debug("lDistCoeffs: " + repr(lDistCoeffs) + " -> lDistCoeffsUpdated: " + repr(lDistCoeffsUpdated))
        logger.debug("rCameraMatrix: " + repr(rCameraMatrix) + " -> rCameraMatrixUpdated: " + repr(rCameraMatrixUpdated))
        logger.debug("rDistCoeffs: " + repr(rDistCoeffs) + " -> rDistCoeffsUpdated: " + repr(rDistCoeffsUpdated))
        logger.debug("R: " + repr(R))
        logger.debug("T: " + repr(T))
        logger.debug("E: " + repr(E))
        logger.debug("F: " + repr(F))
        
        # For debugging only
        imageL = cv2.imread(left_image_paths[0])
        imageR = cv2.imread(right_image_paths[0])
        lUd = cv2.undistort(imageL, lCameraMatrixUpdated, lDistCoeffsUpdated)
        rUd = cv2.undistort(imageR, rCameraMatrixUpdated, rDistCoeffsUpdated)
        # cv2.imwrite('lUd.png', lUd)
        # cv2.imwrite('rUd.png', rUd)
        cv2.imshow('left', imageL)
        cv2.imshow('right', imageR)
        cv2.imshow('left undistorted', lUd)
        cv2.imshow('right undistorted', rUd)
        cv2.waitKey()
        
        # Compute projection matrices
        #https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify
        (leftRectXform, rightRectXform, leftProjMat, rightProjMat, Q, leftRoi, rightRoi) = cv2.stereoRectify(lCameraMatrixUpdated, lDistCoeffsUpdated, rCameraMatrixUpdated, rDistCoeffsUpdated, self._IMAGE_SIZE, R, T)
        logger.debug("leftRectXform : " + repr(leftRectXform ))
        logger.debug("rightRectXform: " + repr(rightRectXform))
        logger.debug("leftProjMat   : " + repr(leftProjMat   ))
        logger.debug("rightProjMat  : " + repr(rightProjMat  ))
        logger.debug("Q: " + repr(Q))
        logger.debug("leftRoi: " + repr(leftRoi))
        logger.debug("rightRoi: " + repr(rightRoi))

        retDict = {
            'minError':minError,
            'leftProjMat':leftProjMat ,
            'rightProjMat':rightProjMat,
        }
        
        return retDict
        
    def find_cal_pattern_in_3space(self, stereo_cal, pair_image_path):
        """
        stereo_cal:  Projection matrices for cv2.triangulatePoints, stored in a dictionary like this:
        {
            'leftProjMat':leftProjMat ,
            'rightProjMat':rightProjMat,
        }
        pair_image_path  : a twople, of the form ("/path/to/one/left/image", "/path/to/one/right/image"),
        
        returns: a list of [x,y,z] coordinates in real-world space, in the form:
            np.array([[   7549.84  , -184252.69  ,   40687.215 ],
                   [   7626.0737, -185671.55  ,   41133.258 ],
                   [   7643.9023, -186005.36  ,   41351.223 ]])
        """
        
        # Find individual dots in all the images
        logger.info("Finding points in left images from pairs")
        all_points_in_3space, all_points_in_left_images = self._find_point_vectors([pair_image_path[0]])
        logger.info("Finding points in right images from pairs")
        all_points_in_3space, all_points_in_right_images = self._find_point_vectors([pair_image_path[1]])
        
        for image_path,points in zip(pair_image_path, [all_points_in_left_images[0], all_points_in_right_images[0]]):
            img = cv2.imread(image_path)
            self._draw_points_on_image(img, points)
            cv2.imshow(image_path, img)
        # logger.debug("Shape: %s"%repr(all_points_in_left_images.shape))
        all_points_in_left_images = all_points_in_left_images[0]
        logger.debug("Shape: %s"%repr(all_points_in_left_images.shape))
        all_points_in_left_images = all_points_in_left_images[:,0,:]
        logger.debug("Shape: %s"%repr(all_points_in_left_images.shape))
        all_points_in_right_images = all_points_in_right_images[0]
        all_points_in_right_images = all_points_in_right_images[:,0,:]
        # Switch from x,y to row,col
        all_points_in_left_images = all_points_in_left_images[:,[1,0]]
        all_points_in_right_images = all_points_in_right_images[:,[1,0]]
        all_points_in_left_images = all_points_in_left_images.transpose()
        logger.debug("Shape: %s"%repr(all_points_in_left_images.shape))
        all_points_in_right_images = all_points_in_right_images.transpose()
        
        logger.debug('all_points_in_left_images: ' + repr(all_points_in_left_images))
        
        points4d = cv2.triangulatePoints(stereo_cal['leftProjMat'], stereo_cal['rightProjMat'], all_points_in_left_images, all_points_in_right_images)
        points3d = cv2.convertPointsFromHomogeneous(points4d.transpose())
        
        logger.debug('points4d: ' + repr(points4d))
        logger.debug('points3d: ' + repr(points3d))
        return points3d[:,0,:]
        

def mark_dots(infilepath, outfilepath, detector):
    """
    Test routine for debugging blob finder params
    """
    image = cv2.imread(infilepath)
    blobs = detector.detect(image)
    print("Found "+str(len(blobs))+" blobs " + infilepath + " -> " + outfilepath)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    annotated = cv2.drawKeypoints(image, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(outfilepath, annotated)

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    sc = StereoCalibrator();
    cal_img_dir = 'test images/2019-10-18 stereo cal images/'
    left_image_names  = [
        'left/left-00001.png',
        'left/left-00002.png',
        'left/left-00003.png',
        'left/left-00004.png',
        'left/left-00005.png',
        'left/left-00006.png',
        # 'left/left-00007.png',
        # 'left/left-00008.png',
        # 'left/left-00009.png',
        'left/left-00010.png',
        'left/left-00011.png',
        'left/left-00012.png',
        'left/left-00013.png',
        'left/left-00014.png',
        'left/left-00015.png',
        'left/left-00019.png',
        # 'left/left-00020.png',
        'left/left-00021.png',
        'left/left-00022.png',
        'left/left-00023.png',
        'left/left-00025.png',
    ]
    right_image_names = [
        'right/right-00001.png',
        'right/right-00002.png',
        # 'right/right-00003.png',
        'right/right-00004.png',
        'right/right-00005.png',
        'right/right-00006.png',
        'right/right-00007.png',
        'right/right-00008.png',
        'right/right-00009.png',
        'right/right-00010.png',
        'right/right-00011.png',
        'right/right-00012.png',
        'right/right-00013.png',
        # 'right/right-00014.png',
        # 'right/right-00015.png',
        'right/right-00019.png',
        'right/right-00020.png',
        'right/right-00021.png',
        'right/right-00022.png',
        'right/right-00023.png',
        # 'right/right-00025.png',
    ]
    pair_image_names = [
        ('left/left-00001.png','right/right-00001.png'),
        ('left/left-00002.png','right/right-00002.png'),
        # ('left/left-00003.png','right/right-00003.png'),
        ('left/left-00004.png','right/right-00004.png'),
        ('left/left-00005.png','right/right-00005.png'),
        ('left/left-00006.png','right/right-00006.png'),
        # ('left/left-00007.png','right/right-00007.png'),
        # ('left/left-00008.png','right/right-00008.png'),
        # ('left/left-00009.png','right/right-00009.png'),
        ('left/left-00010.png','right/right-00010.png'),
        ('left/left-00011.png','right/right-00011.png'),
        ('left/left-00012.png','right/right-00012.png'),
        ('left/left-00013.png','right/right-00013.png'),
        # ('left/left-00014.png','right/right-00014.png'),
        # ('left/left-00015.png','right/right-00015.png'),
        ('left/left-00019.png','right/right-00019.png'),
        # ('left/left-00020.png','right/right-00020.png'),
        ('left/left-00021.png','right/right-00021.png'),
        ('left/left-00022.png','right/right-00022.png'),
        ('left/left-00023.png','right/right-00023.png'),
        # ('left/left-00025.png','right/right-00025.png'),
    ]
    left_cal_images = [cal_img_dir + img for img in left_image_names]
    right_cal_images = [cal_img_dir + img for img in right_image_names]
    pair_cal_images = [(cal_img_dir + pair[0], cal_img_dir + pair[1]) for pair in pair_image_names]
    all_images = left_image_names + right_image_names
    # det = sc.make_detector()
    # for img in all_images:
        # outfile = 'dotted_' + img;
        # mark_dots(cal_img_dir + img, cal_img_dir + outfile, det)
    stereo_cal = sc.find_stereo_pair_calibration(left_cal_images, right_cal_images, pair_cal_images)
    
    output_fn = 'src/stereo_cal.py'
    with open(output_fn, 'w+') as outfile:
        outfile.write('from numpy import array\nstereo_cal = ' + repr(stereo_cal) + '\n')
    logger.info('Done, saved calibration to ' + output_fn)

    # test
    points3d = sc.find_cal_pattern_in_3space(stereo_cal, pair_cal_images[0])
    
    if True:
        # This import registers the 3D projection, but is otherwise unused.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

        import matplotlib.pyplot as plt
        import numpy as np
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points3d[:,0], points3d[:,1], points3d[:,2], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    cv2.waitKey()
