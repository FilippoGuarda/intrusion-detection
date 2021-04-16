import numpy as np
import cv2
import time 



# blob detector parameters

personDetectorParameters = cv2.SimpleBlobDetector_Params()
bookDetectorParameters = cv2.SimpleBlobDetector_Params()


# define params for person detection
personDetectorParameters.filterByArea = True
personDetectorParameters.minArea = 6000 
personDetectorParameters.maxArea = 100000
personDetectorParameters.minDistBetweenBlobs = 0
personDetectorParameters.filterByCircularity = False
personDetectorParameters.filterByColor = True
personDetectorParameters.blobColor = 255
personDetectorParameters.filterByConvexity = False
personDetectorParameters.filterByInertia = False


# define params for book detection
bookDetectorParameters.filterByArea = True
bookDetectorParameters.minArea = 1000 #1000
bookDetectorParameters.maxArea = 4000 #5000
bookDetectorParameters.minDistBetweenBlobs = 0
bookDetectorParameters.filterByCircularity = False
bookDetectorParameters.filterByColor = True
bookDetectorParameters.blobColor = 255
bookDetectorParameters.filterByConvexity = False
bookDetectorParameters.filterByInertia = False

detector_person = cv2.SimpleBlobDetector_create(personDetectorParameters)
detector_book = cv2.SimpleBlobDetector_create(bookDetectorParameters)

# define background subtraction method
# fgbg = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 50, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN(history = 150, dist2Threshold = 150, detectShadows=False)


def change_detection(video_path):
   # previous_frames = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        # Capture frame
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            # Break exit the for loops
            break

        mask = fgbg.apply(frame)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imshow('mask', mask)
        blur=cv2.GaussianBlur(mask,(11,11),0)
        cv2.imshow('Blur', blur)
        ret,thresh = cv2.threshold(blur,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        
       
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        cv2.imshow('opening', opening)
        # dilated = cv2.dilate(opening, None, iterations=2)
        # cv2.imshow('dilated', dilated)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 3)
        cv2.imshow('closing', closing)
        
        
        # TODO: find a way to distinguish if blob was removed from backgound or was added to it
        
        # draw keypoints over grayscale image
        keypoints = detector_person.detect(closing)
        im_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints2 = detector_book.detect(closing)
        im_keypoints2 = cv2.drawKeypoints(im_keypoints, keypoints2, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Video',im_keypoints2)
        time.sleep(0.02)
        if cv2.waitKey(1) == ord('q'):
                break
            
        #TODO: add logging of blobs
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()
    
change_detection('1.avi')