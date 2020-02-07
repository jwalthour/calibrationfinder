#!/usr/bin/python3
"""
Definitions for some simple calibration targets
"""
import logging
logger = logging.getLogger(__name__)
logger.debug('Importing')
import cv2
logger.debug('Done')

printed8x8BlobParams = cv2.SimpleBlobDetector_Params()
printed8x8BlobParams.minThreshold = 0;
printed8x8BlobParams.maxThreshold = 128;
printed8x8BlobParams.filterByArea = True
printed8x8BlobParams.minArea = 5
printed8x8BlobParams.filterByCircularity = True
printed8x8BlobParams.minCircularity = 0.25
printed8x8BlobParams.filterByConvexity = False
printed8x8BlobParams.minConvexity = 0.9
printed8x8BlobParams.maxConvexity = 1
printed8x8BlobParams.filterByInertia = True
printed8x8BlobParams.minInertiaRatio = 0.5
printed8x8BlobParams.minDistBetweenBlobs = 5
printed8x8BlobParams.blobColor = 0




perfboard24x48BlobParams = cv2.SimpleBlobDetector_Params()
perfboard24x48BlobParams.minThreshold = 0;
perfboard24x48BlobParams.maxThreshold = 128;
perfboard24x48BlobParams.filterByArea = True
perfboard24x48BlobParams.minArea = 2
perfboard24x48BlobParams.filterByCircularity = True
perfboard24x48BlobParams.minCircularity = 0.75
perfboard24x48BlobParams.filterByConvexity = False
perfboard24x48BlobParams.minConvexity = 0.5
perfboard24x48BlobParams.maxConvexity = 1
perfboard24x48BlobParams.filterByInertia = True
perfboard24x48BlobParams.minInertiaRatio = 0.5
perfboard24x48BlobParams.minDistBetweenBlobs = 1
perfboard24x48BlobParams.blobColor = 0

# Data structure to import
calibrationTargets = [{
    'desc':'Printed 8x8 grid of dots',
    'simpleBlobDetParams':printed8x8BlobParams,
    'simpleBlobDet': cv2.SimpleBlobDetector_create(printed8x8BlobParams),
    'dims': (8, 8), #rows,cols
    'dotSpacingMm': (25.877, 25.877), #x,y
}, {    
    'desc':'Perfboard 2ft x 4ft target',
    'simpleBlobDetParams':perfboard24x48BlobParams,
    'simpleBlobDet': cv2.SimpleBlobDetector_create(perfboard24x48BlobParams),
    'dims': (24, 48), #rows,cols
    'dotSpacingMm': (25.4, 25.4), #x,y
}]