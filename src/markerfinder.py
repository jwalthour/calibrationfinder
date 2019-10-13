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
            
            masked = self._get_dots(masked)
        # For debugging, we're just gonna return a marked up image for now 
        return masked

    def _get_dots(self, masked_image):
        """
        Locates dot centerpoints in image.  Assumes image is masked down to just one marker's worth of dots.
        
        """
        blobs = self._dot_detector.detect(masked_image)
        print("Found "+str(len(blobs))+" blobs. ")
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        return cv2.drawKeypoints(masked_image, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        

if __name__ == '__main__':
    mf = Markerfinder();
    img = cv2.imread('test images/7 markers on a printout.jpg')
    print('Processing')
    debug_img = mf.find_markers(img)
    print('Saving')
    cv2.imwrite('test.png', debug_img)
    