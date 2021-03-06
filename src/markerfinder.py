#!/usr/bin/python3
print('Importing')
import cv2
import cv2.aruco
import numpy as np

def make_detector():
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

class Markerfinder:
    def __init__(self, marker_dict=None):
        """
        marker_dict : an aruco marker dictionary to use for finding markers
        """
        if marker_dict is None:
            marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
        self._marker_dict = marker_dict
        
        # Courtesy https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        # Setup SimpleBlobDetector parameters.
        self._params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        self._params.minThreshold = 0;
        self._params.maxThreshold = 32;
         
        # Filter by Area.
        self._params.filterByArea = True
        self._params.minArea = 5
         
        # Filter by Circularity
        self._params.filterByCircularity = True
        self._params.minCircularity = 0.25
         
        # Filter by Convexity
        self._params.filterByConvexity = True
        self._params.minConvexity = 0.9
        self._params.maxConvexity = 1
         
        # Filter by Inertia
        self._params.filterByInertia = True
        self._params.minInertiaRatio = 0.5
        
        print("Orig minDistBetweenBlobs: " + str(self._params.minDistBetweenBlobs))
        self._params.minDistBetweenBlobs = 1
        self._params.blobColor = 0
         
        # Create a detector with the parameters
        self._dot_detector = cv2.SimpleBlobDetector_create(self._params)
        
        self._cal_target_dot_det = make_detector()
        
        # Set up the calibration pattern 
        self.CAL_PATTERN_DIMS = (8, 8)  # in dots
        self.CAL_DOT_SPACING = (25.8, 25.8)  # in mm
        self._IMAGE_SIZE = (800,600)  # in px
        self._cal_3space_pattern = []
        for x in range(0, self.CAL_PATTERN_DIMS[0]):
            for y in range(0, self.CAL_PATTERN_DIMS[1]):
                self._cal_3space_pattern += [(x * self.CAL_DOT_SPACING[0], y * self.CAL_DOT_SPACING[1], 0)]
        self._cal_3space_pattern
    
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
        lCameraMatrix, lDistCoeffs = mf.find_single_cam_calibration(left_image_paths)
        rCameraMatrix, rDistCoeffs = mf.find_single_cam_calibration(right_image_paths)
        
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
        

    def find_markers_single_image(self, image):
        """
        Locates markers that consist of an aruco marker surrounded by 20 dots
        
        image : An opencv image
        returns : A dictionary, whose keys are aruco marker IDs, and values are twoples indicating marker centerpoint
        """
        corners, ids, rejects = cv2.aruco.detectMarkers(image, self._marker_dict)

        markedup = cv2.aruco.drawDetectedMarkers(image, corners, ids)

        print("corners: " + repr(corners))
        markers = {}
        for i in range(0, len(ids)):
            # print(str(i) + ": corners[i]: " + repr(corners[i]))
            # print(str(i) + ": vector_ul_lr: " + repr(vector_ul_lr))
            # print(str(i) + ": 0.25 * vector_ul_lr: " + repr(0.25 * vector_ul_lr))
            # print(str(i) + ": corners[i][0][2] + 0.25 * vector_ul_lr: " + repr(corners[i][0][2] + 0.25 * vector_ul_lr))
            # Compute a box 25% wider in every dimension.
            # I sorted this out in longform and then condensed it.  Sorry.
            grown_box = np.array([corners[i][0][(c + 2) % 4] + 0.25 * (corners[i][0][(c + 2) % 4] - corners[i][0][c]) for c in range(0,4)], dtype=np.int32)
            # print(str(i) + ": grown_box: " + repr(grown_box))
            poly_pts = grown_box.reshape((-1,1,2))
            
            # Create a mask that will select just the dots
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, poly_pts, 1, 0)
            cv2.fillConvexPoly(mask, corners[i].astype(np.int32), 0, 0)
            masked = cv2.bitwise_and(image, image, mask=mask)
            cv2.polylines(markedup, [poly_pts], True, (0,255,0))
            
            dots = self._get_dots(masked)
            # print("Dots at: " + repr(dots))
        # For debugging, we're just gonna return a marked up image for now 
        # return masked
        return dots

    def _get_dots(self, masked_image):
        """
        Locates dot centerpoints in image.  Assumes image is masked down to just one marker's worth of dots.
        returns : list of twoples, representing (x,y) coordinates in image of the 
        """
        blobs = self._dot_detector.detect(masked_image)
        print("Found "+str(len(blobs))+" blobs. ")
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        # return cv2.drawKeypoints(masked_image, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return [blob.pt for blob in blobs]


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
    from stereo_cal import stereo_cal
    mf = Markerfinder();
    marker_img_dir = 'test images/2019-10-18 stereo cal images/'
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
    det = make_detector()
    for img in all_images:
        outfile = 'dotted_' + img;
        mark_dots(cal_img_dir + img, cal_img_dir + outfile, det)
    mf.find_stereo_pair_calibration(left_cal_images, right_cal_images, pair_cal_images)
    mf.find_markers_single_image(img)
    