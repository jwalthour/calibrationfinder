#!/usr/bin/python3
print('Importing')
import cv2
import cv2.aruco
import numpy as np

def find_markers(image, marker_dict=None):
    """
    Locates markers that consist of an aruco marker surrounded by 20 dots
    
    image : An opencv image
    marker_dict : an aruco marker dictionary to use for finding markers
    returns : A dictionary, whose keys are aruco marker IDs, and values are twoples indicating marker centerpoint
    """
    if marker_dict is None:
        marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
    corners, ids, rejects = cv2.aruco.detectMarkers(image, marker_dict)

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
    # For debugging, we're just gonna return a marked up image for now 
    return masked


if __name__ == '__main__':
    img = cv2.imread('test images/7 markers on a printout.jpg')
    print('Processing')
    debug_img = find_markers(img)
    print('Saving')
    cv2.imwrite('test.png', debug_img)
    