#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 22:15:48 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 14:11:41 2021

@author: sora
"""
import numpy as np
import cv2
import time


def L1(img1, img2):
    diff = np.abs(img1 - img2)
    print(len(img1.shape))
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        diff = np.sum(diff, axis=-1)
    return diff


def L2(img1, img2):
    sq_dist = (img1 - img2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff


def Linf(F1, F2):
    diff = np.abs(F1 - F2)
    if F1.shape[-1] == 3 and len(F1.shape) == 3:
        diff = np.max(diff, axis=-1)
    return diff


def twoframedifference(frame, previousframe, distance_type, threshold):
    distance = distance_type(frame, previousframe)
    maskbool = distance > threshold
    mask = maskbool.astype(np.uint8) * 255
    return [mask]

def threeframedifference(frame,prev1,prev2, distance_type, threshold):
    if distance_type=="L1":
        [maskbool,mask1]=twoframedifference(frame,prev1, distance_type, threshold)
        if len(prev2)!=0:
            [maskbool1,mask2]=twoframedifference(prev1,prev2, distance_type, threshold)
    mask=np.logical_and(maskbool, maskbool1)
    prev1[np.logical_not(mask)]=np.array([255,255,255])
    mask=mask.astype(np.uint8)*255



def pfm(hist):
    total_pixel = np.sum(hist)
    pfm = []
    for i in range(256):
        pfm_i = np.sum(hist[:i]) / total_pixel
        pfm.append(pfm_i)
    return np.asarray(pfm)


Rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean


def background_initialization(bg, n, cap, count):
    n=n*3
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret or not frame is None:
            # Release the Video if ret is false
            if (count % 3 != 0): 
                hist,bins = np.histogram(frame.flatten(),256,[0,256])
                eq_op = pfm(hist)*255
                bg.append(eq_op)
            count += 1
        else:
            break
    cap.release()
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    cv2.destroyAllWindows()
    return [bg_inter, count]


def background_update(bg1,bg, prev_bg, alfa):
    bg1 = (1 - alfa) * prev_bg + alfa * bg
    return bg1


###Define change detection parameters
thr = 25
distance = L2
bg = []
bg1=[]
frame=[]
N_frames = 30 # then refresh


def sobel(img):
    dst1 = cv2.Sobel(img, -1, 1, 0, 3)
    dst2 = cv2.Sobel(img, -1, 0, 1, 3)
    sob = np.maximum(dst1, dst2)
    return sob


# blob detector parameters

personDetectorParameters = cv2.SimpleBlobDetector_Params()
bookDetectorParameters = cv2.SimpleBlobDetector_Params()

# define params for person detection
personDetectorParameters.filterByArea = True
personDetectorParameters.minArea = 5000  # 5000
personDetectorParameters.maxArea = 100000
personDetectorParameters.minDistBetweenBlobs = 0
personDetectorParameters.filterByCircularity = False
personDetectorParameters.filterByColor = True
personDetectorParameters.blobColor = 255
personDetectorParameters.filterByConvexity = False
personDetectorParameters.filterByInertia = False


# define params for book detection
bookDetectorParameters.filterByArea = True
bookDetectorParameters.minArea = 500  # 1000
bookDetectorParameters.maxArea = 3000  # 5000
bookDetectorParameters.minDistBetweenBlobs = 0
bookDetectorParameters.filterByCircularity = False
bookDetectorParameters.filterByColor = True
bookDetectorParameters.blobColor = 255
bookDetectorParameters.filterByConvexity = False
bookDetectorParameters.filterByInertia = False


detector_person = cv2.SimpleBlobDetector_create(personDetectorParameters)
detector_book = cv2.SimpleBlobDetector_create(bookDetectorParameters)
cap = cv2.VideoCapture('1.avi')
count = 0

# computation of the background
[bg, count] = background_initialization(bg, N_frames, cap, count)


#fgbg = cv2.createBackgroundSubtractorKNN(1,10,False)


def change_detection(video_path, bg, threshold,frame):
    cap = cv2.VideoCapture(video_path)
    prevhist = 0
    cond = False
    while (cap.isOpened()):
        # Capture frame
        ret, frame = cap.read()
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            # Break exit the for loops
            break
        #Convert to grayscale and blur frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Compute background suptraction
        mask = (1-bg > threshold)
        mask = mask.astype(np.uint8) * 255


        #mask= fgbg.apply(gray)
        cv2.imshow('mask', mask)

        blur=cv2.GaussianBlur(mask,(11,11),0)
        #cv2.imshow('Blur', blur)
        ret,thresh = cv2.threshold(blur,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('thresh', thresh)
        
       
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        #cv2.imshow('opening', opening)
        # dilated = cv2.dilate(opening, None, iterations=2)
        # cv2.imshow('dilated', dilated)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 3)
        #cv2.imshow('closing', closing)

        hist, bins = np.histogram(closing.flatten(), 256, [0, 256])
        #update background when ligth changes
        if (cond==True and hist[255] > 1.1*prevhist) :
            #bg = background_update(bg1, gray, bg, 0.05)
            print('change_updated')
        prevhist=hist[255]
 
        #keypoints = detector_person.detect(dilated)
        #im_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
        #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        keypoints2 = detector_book.detect(closing)
        #use keypoints to update background
        frame = cv2.drawKeypoints(gray, keypoints2, np.array([]), (255, 0, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Video', frame)
        _, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        blob_count = len(contours)
        for i, cnt in enumerate(contours):
            # if the size of the contour is greater than a threshold
            if cv2.contourArea(cnt) < 6000:
                continue
            #elif cv2.contourArea(cnt) < 2000:
            #    cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)  # if >0 shows contour
            else:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                #cv2.drawContours(im_keypoints2, [cnt], 0, (255, 255, 255), 3)

        cv2.imshow('contours', frame)
        #if (blob_count<1):
            #bg = background_update(bg1, gray, bg, 0.7)
        #cv2.resizeWindow('contours', 500, 500)
        time.sleep(0.02)
        cond = True
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr, frame)
