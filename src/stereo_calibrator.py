#!/usr/bin/python3
print('Importing')
import cv2
import cv2.aruco
import numpy as np

class StereoCalibrator:
    def __init__(self):
        """
        """
        self._cal_target_dot_det = self.make_detector()
        
        # Set up the calibration pattern 
        self.CAL_PATTERN_DIMS = (8, 8)  # in dots
        self.CAL_DOT_SPACING_MM = (25.8, 25.8)  # in mm
        self._IMAGE_SIZE = (800,600)  # in px
        self._cal_3space_pattern = []
        for x in range(0, self.CAL_PATTERN_DIMS[0]):
            for y in range(0, self.CAL_PATTERN_DIMS[1]):
                self._cal_3space_pattern += [(x * self.CAL_DOT_SPACING_MM[0], y * self.CAL_DOT_SPACING_MM[1], 0)]
        self._cal_3space_pattern
    
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
        
        print("Orig minDistBetweenBlobs: " + str(parms.minDistBetweenBlobs))
        parms.minDistBetweenBlobs = 5
        parms.blobColor = 0
         
        # Create a detector with the parameters
        return cv2.SimpleBlobDetector_create(parms)

    def find_single_cam_calibration(self, image_paths):
        """
        images : list of image file paths
        """
        all_points_in_3space, all_points_in_images = self._find_point_vectors(image_paths)
        if len(all_points_in_3space) > 0:
            # print("np.array(all_points_in_3space) = " + repr(np.array(all_points_in_3space)))
            all_points_in_3space = np.array(all_points_in_3space, dtype=np.float32)
            # print("all_points_in_3space = " + str(all_points_in_3space))
            found,cameraMatrix,distCoeffs,rvecs,tvecs = cv2.calibrateCamera(all_points_in_3space, all_points_in_images, self._IMAGE_SIZE, np.array([]), np.array([]))
            # print("found: " + repr(found) + ",\n cameraMatrix: " + repr(cameraMatrix) + ",\n distCoeffs: " + repr(distCoeffs) + ",\n rvecs: " + repr(rvecs) + ",\n tvecs: " + repr(tvecs))
        return cameraMatrix,distCoeffs
    
    def _find_point_vectors(self, image_paths):
        all_points_in_images = []
        all_points_in_3space = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            # print("img : " + repr(img))
            points = np.array([[]])
            found,points = cv2.findCirclesGrid(img, self.CAL_PATTERN_DIMS, points, cv2.CALIB_CB_SYMMETRIC_GRID, self._cal_target_dot_det)
            print(("Found " + str(len(points)) + " cal points in " + image_path) if found else "No cal pattern found in " + image_path)
            if found:
                all_points_in_images += [points]
                all_points_in_3space += [self._cal_3space_pattern]
        return all_points_in_3space, all_points_in_images
    
    def find_stereo_pair_calibration(self, left_image_paths, right_image_paths, pair_image_paths):
        """
        
        left_image_paths  : list of strings, each of which is a path to an image from the left camera
        right_image_paths : list of strings, each of which is a path to an image from the right camera
        pair_image_paths  : list of twoples, of the form ("/path/to/one/left/image", "/path/to/one/right/image"),
        
        """
        # First must calibrate individual cameras
        lCameraMatrix, lDistCoeffs = self.find_single_cam_calibration(left_image_paths)
        rCameraMatrix, rDistCoeffs = self.find_single_cam_calibration(right_image_paths)
        
        # redefine these 
        left_image_paths = [pair[0] for pair in pair_cal_images]
        right_image_paths = [pair[1] for pair in pair_cal_images]
        
        # Find individual dots in all the images
        all_points_in_3space, all_points_in_left_images = self._find_point_vectors(left_image_paths)
        all_points_in_3space, all_points_in_right_images = self._find_point_vectors(right_image_paths)
        all_points_in_3space = np.array(all_points_in_3space, dtype=np.float32)
        # print("all_points_in_3space: " + repr(all_points_in_3space))
        # print("all_points_in_left_images: " + repr(all_points_in_left_images))
        # print("all_points_in_right_images: " + repr(all_points_in_right_images))
        # print("self._IMAGE_SIZE: " + repr(self._IMAGE_SIZE))
        print("len(all_points_in_3space): " + str(len(all_points_in_3space)))
        print("len(all_points_in_left_images): " + str(len(all_points_in_left_images)))
        print("len(all_points_in_right_images): " + str(len(all_points_in_right_images)))
        retval, lCameraMatrix, lDistCoeffs, rCameraMatrix, rDistCoeffs, R, T, E, F = cv2.stereoCalibrate(all_points_in_3space, all_points_in_left_images, all_points_in_right_images, lCameraMatrix, lDistCoeffs, rCameraMatrix, rDistCoeffs, self._IMAGE_SIZE)
        print("retval: " + repr(retval))
        print("lCameraMatrix: " + repr(lCameraMatrix))
        print("lDistCoeffs: " + repr(lDistCoeffs))
        print("rCameraMatrix: " + repr(rCameraMatrix))
        print("rDistCoeffs: " + repr(rDistCoeffs))
        print("R: " + repr(R))
        print("T: " + repr(T))
        print("E: " + repr(E))
        print("F: " + repr(F))
        
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
    det = sc.make_detector()
    for img in all_images:
        outfile = 'dotted_' + img;
        mark_dots(cal_img_dir + img, cal_img_dir + outfile, det)
    sc.find_stereo_pair_calibration(left_cal_images, right_cal_images, pair_cal_images)
