import numpy as np
import cv2
import time 


f = open("detected_log.txt", "w+") 



def skip_background(contours, frame, original_contour, shift1, shift2, index, thresh = 30):
    #ignore contours that are part of the background
    
    
    cv2.drawContours(original_contour, contours, index, 255, -1)

    # take two shifted contours, add them and mask using original contours to obtain internal contour
    cv2.drawContours(shift1, contours, index, 255, 10, offset=(5,0))
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(-5,0))
    # bitwise or sums white pixels and masks with previously found contour, get internal contour
    internal_contour = cv2.bitwise_or(shift1, shift2, mask = original_contour)
    # get external contour, 255-original contour creates inverted mask
    external_contour = cv2.bitwise_or(shift1, shift2, mask = 255 - original_contour)
    external_median = np.median(frame[external_contour > 0], overwrite_input=True)
    internal_median = np.median(frame[internal_contour > 0], overwrite_input=True)
    if np.abs(external_median - internal_median) < thresh:
        return True
    
def object_detector(contours, index, mask, color_mask, frame_number): 
    #detect, classify and log objects moving in the frame

    
    #define area and perimeter
    area = cv2.contourArea(contours[index])
    perimeter = cv2.arcLength(contours[index], True)
    
    #detect person
    if (6000 < area < 20000) and (perimeter  < 900):
        cv2.drawContours(mask, contours, index, 255, -1)
        cv2.drawContours(color_mask, contours, index, [0,0,255], -1)
        f.write("frame %d, detected person, blob area: %d, blob perimeter: %d\r\n"% (frame_number, area, perimeter))
        
    #detect book    
    if (600 < area < 3000) and (perimeter < 200):
        cv2.drawContours(mask, contours, index, 255, -1)
        cv2.drawContours(color_mask, contours, index, [255,0,0], -1)
        f.write("frame %d, detected book, blob area: %d, blob perimeter: %d\r\n"% (frame_number, area, perimeter))

# define background subtraction method
# fgbg = cv2.createBackgroundSubtractorMOG2(history = 500, varThreshold = 50, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN(history = 150, dist2Threshold = 150, detectShadows=True)



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
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        
       
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        cv2.imshow('opening', opening)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 2)
        cv2.imshow('closing', closing)
        
        # find a way to distinguish if blob was removed from backgound or was added to it
        contours, hierarchy = cv2.findContours(closing.astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # variable initialization as zero arrays
        image_external = np.zeros(closing.shape, np.uint8)
        colored_contours = np.zeros(frame.shape)
        original_contour = np.zeros(closing.shape, np.uint8)
        shift2 = np.zeros(closing.shape, np.uint8)
        shift1 = np.zeros(closing.shape, np.uint8)
        
        
        for i in range(len(contours)):
            
            # reducing treshold augments detection capability, but more false positives
            if skip_background(contours, frame, original_contour, shift1, shift2, i, 25):
                continue
            else:
                # generate log of detected objects per frame
                object_detector(contours, i, image_external, colored_contours, frame_number)
        
          
        cv2.imshow('contours', image_external)
        cv2.imshow('contours', colored_contours)
        
        #visualize masked image
        masked_image = np.copy(frame)
        masked_image[image_external < 50] = 0
        cv2.imshow('masked image', masked_image)


        time.sleep(0.02)
        if cv2.waitKey(1) == ord('q'):
                break
        
        frame_number += 1
            
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()
    
change_detection('1.avi')
f.close()