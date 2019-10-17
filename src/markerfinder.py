#!/usr/bin/python3
print('Importing')
import cv2
import cv2.aruco
import numpy as np

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
        self._params.minArea = 100
         
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
         
        # Create a detector with the parameters
        self._dot_detector = cv2.SimpleBlobDetector_create(self._params)
        
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
        all_points_in_3space, all_points_in_images = _find_point_vectors(image_paths)
        if len(all_points_in_3space) > 0:
            # print("np.array(all_points_in_3space) = " + repr(np.array(all_points_in_3space)))
            all_points_in_3space = np.array(all_points_in_3space, dtype=np.float32)
            # print("all_points_in_3space = " + str(all_points_in_3space))
            found,cameraMatrix,distCoeffs,rvecs,tvecs = cv2.calibrateCamera(all_points_in_3space, all_points_in_images, self._IMAGE_SIZE, cameraMatrix, distCoeffs)
            # print("found: " + repr(found) + ",\n cameraMatrix: " + repr(cameraMatrix) + ",\n distCoeffs: " + repr(distCoeffs) + ",\n rvecs: " + repr(rvecs) + ",\n tvecs: " + repr(tvecs))
        return cameraMatrix,distCoeffs
    
    def _find_point_vectors(self, image_paths):
        all_points_in_images = []
        all_points_in_3space = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            # print("img : " + repr(img))
            found,points = cv2.findCirclesGrid(img, self.CAL_PATTERN_DIMS)
            print(("Found " + str(len(points)) + " cal points in " + image_path) if found else "No cal pattern found in " + image_path)
            if found:
                all_points_in_images += [points]
                all_points_in_3space += [self._cal_3space_pattern]
        return all_points_in_3space, all_points_in_images
    
    def find_stereo_pair_calibration(self, left_image_paths, right_image_paths):
        # lCameraMatrix, lDistCoeffs = mf.find_single_cam_calibration(left_image_paths)
        # rCameraMatrix, rDistCoeffs = mf.find_single_cam_calibration(right_image_paths)
        all_points_in_3space, all_points_in_left_images = self._find_point_vectors(left_image_paths)
        all_points_in_3space, all_points_in_right_images = self._find_point_vectors(right_image_paths)
        all_points_in_3space = np.array(all_points_in_3space, dtype=np.float32)
        # print("all_points_in_3space: " + repr(all_points_in_3space))
        # print("all_points_in_left_images: " + repr(all_points_in_left_images))
        # print("all_points_in_right_images: " + repr(all_points_in_right_images))
        # print("self._IMAGE_SIZE: " + repr(self._IMAGE_SIZE))
        retval, lCameraMatrix, lDistCoeffs, rCameraMatrix, rDistCoeffs, R, T, E, F = cv2.stereoCalibrate(all_points_in_3space, all_points_in_left_images, all_points_in_right_images, np.array([]), np.array([[]]), np.array([]), np.array([[]]), self._IMAGE_SIZE)
        print("retval: " + repr(retval))
        print("lCameraMatrix: " + repr(lCameraMatrix))
        print("lDistCoeffs: " + repr(lDistCoeffs))
        print("rCameraMatrix: " + repr(rCameraMatrix))
        print("rDistCoeffs: " + repr(rDistCoeffs))
        print("R: " + repr(R))
        print("T: " + repr(T))
        print("E: " + repr(E))
        print("F: " + repr(F))
        

    def find_markers(self, image):
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
        return masked

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
        

if __name__ == '__main__':
    mf = Markerfinder();
    cal_img_dir = 'test images/2019-10-13 stereo cal images/'
    left_cal_images  = [cal_img_dir + fn for fn in ['0r.jpg', '1r.jpg', '2r.jpg', '3r.jpg', '4r.jpg', '5r.jpg']]
    right_cal_images = [cal_img_dir + fn for fn in ['0r.jpg', '1r.jpg', '2r.jpg', '3r.jpg', '4r.jpg', '5r.jpg']]
    mf.find_stereo_pair_calibration(left_cal_images, right_cal_images)
    # img = cv2.imread('test images/7 markers on a printout.jpg')
    # print('Processing')
    # debug_img = mf.find_markers(img)
    # print('Saving')
    # cv2.imwrite('test.png', debug_img)
    