import cv2

def readVideoStream(url):
    cap = cv2.VideoCapture(url)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame.")
            return
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) == 27:
            return

if __name__ == '__main__':
    # Make a file that defines these two variables
    from auth import USERNAME,PASSWORD
    # Left camera in my setup.  Replace with yours.
    urlPattern = 'http://%s:%s@192.168.0.253/mjpg/video.mjpg'
    url = urlPattern%(USERNAME,PASSWORD)
    readVideoStream(url)

    