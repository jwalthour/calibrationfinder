#!/usr/bin/python3
print('Importing')
import cv2
import cv2.aruco

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
        # "Grow" the aruco marker frame by 25%, which is the width of the margin of dots
        vector_ul_lr = corners[i][0][2] - corners[i][0][0]
        # print(str(i) + ": vector_ul_lr: " + repr(vector_ul_lr))
        # print(str(i) + ": 0.25 * vector_ul_lr: " + repr(0.25 * vector_ul_lr))
        # print(str(i) + ": corners[i][0][2] + 0.25 * vector_ul_lr: " + repr(corners[i][0][2] + 0.25 * vector_ul_lr))
        grown_box = corners[i][0][2] + 0.25 * vector_ul_lr
        print(str(i) + ": grown_box: " + repr(grown_box))
        
    # For debugging, we're just gonna return a marked up image for now 
    return markedup


if __name__ == '__main__':
    img = cv2.imread('test images/7 markers on a printout.jpg')
    print('Processing')
    debug_img = find_markers(img)
    print('Saving')
    cv2.imwrite('test.png', debug_img)
    