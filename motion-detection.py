import numpy as np
import cv2
import time 



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



def skip_background(contours, frame, original_contour, shift1, shift2, index, thresh = 30):
    #ignore contours that are part of the background
    
    
    cv2.drawContours(original_contour, contours, index, 255, -1)

    # take two shifted contours, add them and mask using original contours to obtain internal contour
    cv2.drawContours(shift1, contours, index, 255, 10, offset=(5,0))
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(-5,0))
    internal_contour = cv2.bitwise_or(shift1, shift2, mask = original_contour)
    # get external contour
    external_contour = cv2.bitwise_or(shift1, shift2, mask = 255 - original_contour)
    external_median = np.median(frame[external_contour > 0], overwrite_input=True)
    internal_median = np.median(frame[internal_contour > 0], overwrite_input=True)
    if np.abs(external_median - internal_median) < thresh:
        return True
    
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
fgbg = cv2.createBackgroundSubtractorKNN(history = 150, dist2Threshold = 200, detectShadows=True)



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
        cv2.imshow('mask', mask)
        blur=cv2.GaussianBlur(mask,(5,5),0)
        cv2.imshow('Blur', blur)
        ret,thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        
       
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        cv2.imshow('opening', opening)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 2)
        cv2.imshow('closing', closing)
        
        # TODO: find a way to distinguish if blob was removed from backgound or was added to it
        contours, hierarchy = cv2.findContours(closing.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_external = np.zeros(closing.shape, np.uint8)
        colored_contours = np.zeros(frame.shape)
        original_contour = np.zeros(closing.shape, np.uint8)
        shift2 = np.zeros(closing.shape, np.uint8)
        shift1 = np.zeros(closing.shape, np.uint8)
        
        
        for i in range(len(contours)):
            
            if skip_background(contours, frame, original_contour, shift1, shift2, i, 20):
                continue
            else:
                object_detector(contours, i, image_external, colored_contours)
        
          
        # cv2.imshow('contours', image_external)
        # cv2.imshow('contours', colored_contours)
        
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
        # cv2.imshow('Video',im_keypoints2)
        time.sleep(0.02)
        if cv2.waitKey(1) == ord('q'):
                break
        
        frame_number += 1
            
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()
    
change_detection('1.avi')