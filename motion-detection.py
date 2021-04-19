import numpy as np
import cv2
import time 
import random



# blob detector parameters

personDetectorParameters = cv2.SimpleBlobDetector_Params()
bookDetectorParameters = cv2.SimpleBlobDetector_Params()


# define params for person detection
personDetectorParameters.filterByArea = True
personDetectorParameters.minArea = 6000 
personDetectorParameters.maxArea = 10000
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

def object_detector(contours, index, mask, color_mask): 
    #detect, classify and log objects moving in the frame
    
    #detect person
    if (6000 < cv2.contourArea(contours[index]) < 20000) and (cv2.arcLength(contours[index], True) < 900):
        cv2.drawContours(mask, contours, index, 255, -1)
        cv2.drawContours(color_mask, contours, index, [0,0,255], -1)
    #detect book    
    if (500 < cv2.contourArea(contours[index]) < 5000):
        cv2.drawContours(mask, contours, index, 255, -1)
        cv2.drawContours(color_mask, contours, index, [255,0,0], -1)

# define background subtraction method
# fgbg = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 50, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN(history = 150, dist2Threshold = 150, detectShadows=False)

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)


def change_detection(video_path):
   # previous_frames = []
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
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
        # cv2.imshow('mask', mask)
        blur=cv2.GaussianBlur(mask,(11,11),0)
        # cv2.imshow('Blur', blur)
        ret,thresh = cv2.threshold(blur,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cv2.imshow('thresh', thresh)
        
       
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        # cv2.imshow('opening', opening)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 2)
        # cv2.imshow('closing', closing)
        eroded = cv2.erode(closing, (5, 5), iterations = 2)
        
        # TODO: find a way to distinguish if blob was removed from backgound or was added to it
        contours, hierarchy = cv2.findContours(closing.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        eroded_contours, hierarchy = cv2.findContours(eroded.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_external = np.zeros(closing.shape, np.uint8)
        colored_contours = np.zeros(frame.shape)
        
        
        for i in range(len(contours)):
            
            #ignore contours that are part of the background
            original_contour = shifted_contour = np.zeros(closing.shape, np.uint8)
            cv2.drawContours(original_contour, contours, i, 255, 3)
            cv2.imshow('original', original_contour)
            for j in range(len(eroded_contours)):
                cv2.drawContours(shifted_contour, eroded_contours, j, 255, 3)
                cv2.imshow('shifted', shifted_contour)
                if 
            
            object_detector(contours, i, image_external, colored_contours)
                
        cv2.imshow('contours', image_external)
        cv2.imshow('contours', colored_contours)
        
        #visualize masked image
        masked_image = np.copy(frame)
        masked_image[image_external < 50] = 0
        cv2.imshow('masked image', masked_image)
        #TODO: generate log of detected objects per frame
        
        # draw keypoints over grayscale image
        keypoints = detector_person.detect(image_external)
        im_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints2 = detector_book.detect(image_external)
        im_keypoints2 = cv2.drawKeypoints(im_keypoints, keypoints2, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Video',im_keypoints2)
        time.sleep(0.02)
        if cv2.waitKey(1) == ord('q'):
                break
        
        frame_number += 1
            
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()
    
change_detection('1.avi')