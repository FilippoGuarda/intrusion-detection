#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:26:18 2021

@author: sora
"""

"""
Created on Sat Apr 17 14:11:41 2021

@author: sora
"""
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

def L1(img1, img2):
    diff = np.abs(img1 - img2)
    print(len(img1.shape))
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        diff = np.sum(diff, axis=-1)
    return diff


def L2_norm(img1, img2):
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    newimg1 = (img1-mean1)/np.std(img1)
    newimg2 = (img2-mean2)/np.std(img2)
    sq_dist = (newimg1 - newimg2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff


def Linf(F1, F2):
    diff = np.abs(F1 - F2)
    if F1.shape[-1] == 3 and len(F1.shape) == 3:
        diff = np.max(diff, axis=-1)
    return diff

def two_frame_difference(frame, previous_frame, distance, th):
    dist = distance(frame, previous_frame)
    mask = dist > th
    return mask

def three_frame_difference(frame, previous_frames, distance, th):
    masks = []
    masks.append(two_frame_difference(frame, previous_frames[1], distance, th))
    if len(previous_frames) > 1:
        masks.append(two_frame_difference(previous_frames[1], previous_frames[0], distance, th))
    and_mask = np.prod(masks,axis=0)
    and_mask = and_mask.astype(np.uint8) * 255
    return and_mask


# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean


def selective_background_initialization3(bg, n, cap, count2):
    thresh=0.085
    selective=[]
    previous_frames = []
    n=n*2
    count = 0
    while cap.isOpened() and count2<n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if not ret or frame is None:
            # Release the Video if ret is false
            cap.release()
            print("Released Video Resource")
            break
        if (count2 % 2 != 0): 
            if count < 2 :
                # initialize previous frames properly
                previous_frames.append(frame.astype(float))
            else:
                frame = frame.astype(float)
                mask = three_frame_difference(frame, previous_frames, distance, thresh)
                mask = mask.astype(np.uint8)
#                cv2.imshow('gf', mask.astype(np.uint8))
                # update previous frames
                sopen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
#                cv2.imshow('gf2', sopen)
#                sdilate = cv2.dilate(sopen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                sdilate = cv2.morphologyEx(sopen, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=2)
#                cv2.imshow('gf3', sdilate)
                sinv = 255 - sdilate
                sbg = np.copy(frame)
                sbg[np.logical_not(sinv)] = np.asarray(0)
 #               cv2.imshow('gf4', sbg)
                selective.append(sbg)
#                previous_frames=frame
                previous_frames.pop(0)
                previous_frames.append(frame)
            count +=1
        count2 += 1
        # Stop playing when entered 'q' from keyboard
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    bg_inter = np.stack(selective, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    plt.imshow(bg_inter.astype(np.uint8), cmap='gray', vmin=0, vmax=255)    
    cv2.destroyAllWindows()
    return bg_inter


def background_initialization(bg, n, cap, count):
    n=n*2
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret and not frame is None:
            if (count % 2 != 0): 
                bg.append(frame)
            count += 1
        else:
            break
    cap.release()
    b = bg.copy()
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    cv2.destroyAllWindows()
    return [b, bg_inter, count]

def selective_background_update(bg1, frame, prev_bg, alfa,closing):
    frame[np.logical_not(closing)] = np.asarray(0)
    bg2 = np.copy(prev_bg)
    bg3 = np.copy(prev_bg)
    prev_bg = prev_bg.astype(np.uint8)
    prev_bg[np.logical_not(closing)] = np.asarray(0)
    bg3[closing==255] = np.asarray(0)
    bg2= (1 - alfa) * prev_bg + alfa * frame +bg3
    bg1 = np.copy(bg2)
    return bg1

def skip_background(contours, frame, final, shift1, shift2, index, thresh):
   # ignore contours that are part of the background
    # take two shifted contours, add them and mask using original contours to obtain internal contour
    #print(index)
    cv2.drawContours(shift1, contours, index, 255, 10, offset=(0, 0))
    shift1=cv2.erode(shift1,kernel,iterations=5)
    #cv2.imshow('internal',shift1)
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(0, 0))
    #shift2=cv2.dilate(shift2, kernel, iterations=5)
    shift2=shift2-final
    #shift2=cv2.erode(shift2,kernel,iterations=4)
    #cv2.imshow('external', shift2)
    external_median =(frame[shift1 > 0])
    hist = cv2.calcHist([external_median], [0], None, [256], [0, 256])
    internal_median =(frame[shift2 > 0])
    hist1 = cv2.calcHist([internal_median], [0], None, [256], [0, 256])
    #print('internal %d',internal_median)
    compare= cv2.compareHist(hist, hist1, cv2.HISTCMP_CORREL)
    #print(compare)
    if compare > thresh:
        return True
###Define change detection parameters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thr = 30
distance = L2_norm
bg = []
b=[]
bg1=[]
bg2=[]
frame=[]
N_frames = 35 # then refresh
# blob detector parameters

cap = cv2.VideoCapture('1.avi')
count = 0

# computation of the background
bg = selective_background_initialization3(bg, N_frames, cap, count)

#fgbg = cv2.createBackgroundSubtractorKNN(1,10,False)
file = open("detected_log.txt", "w+")

def change_detection(video_path, bg, threshold,frame,b):
    cap = cv2.VideoCapture(video_path)
    prevhist = 0
    frame_number = 0
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
        cv2.imshow('bg', bg.astype(np.uint8))
  
        #Compute background suptraction
        mask = (distance(gray, bg) > 0.5)
        mask = mask.astype(np.uint8) * 255
        cv2.imshow('mask', mask)
        blur=cv2.GaussianBlur(mask,(5,5),0)
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        dilated = cv2.dilate(opening,  cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)), iterations=3)
        closing=dilated
        mask6 =  (distance(gray, bg) > 0.25)
        mask6 = mask6.astype(np.uint8) * 255
        mask6[np.logical_not(closing)]=np.asarray(0)
        blur6=cv2.GaussianBlur(mask6,(5,5),3)
        ret6,thresh6 = cv2.threshold(blur6,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('mask6', mask6)
        
        opening2 = cv2.morphologyEx(thresh6, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 2)
        out = closing2
        im2=gray.copy()
        closing4=cv2.dilate(closing2, None, iterations=2)
        closing3=255-closing4
        im2[np.logical_not(closing3)] = np.asarray(0)
      
        hist, bins = np.histogram(thresh6.flatten(), 256, [0, 256])
        #update background when ligth changes
       
        _, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        final = out
       
        if (hist[255] < 0.2*prevhist):
            bg = selective_background_update(bg1, gray, bg, 0.3, closing3)

        shift2 = np.zeros(final.shape, np.uint8)
        shift1 = np.zeros(final.shape, np.uint8)

        for i, cnt in enumerate(contours):
             #person detector
             if cv2.contourArea(cnt)>4500:
                #draw person in blue
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                file.write("frame %d, detected person, blob area: %d, blob perimeter: %d\r\n"% (frame_number, area, perimeter))
                cv2.drawContours(frame, contours,i,[255, 0, 0], -1)

        for j, cnt in enumerate(contours):

            if (450 < cv2.contourArea(contours[j]) < 1350):
                if skip_background(contours, frame, final , shift1, shift2, j, 0.9) == True:
                #draw false object in red
                    area1 = cv2.contourArea(cnt)
                    perimeter1 = cv2.arcLength(cnt, True)
                    file.write("frame %d, detected FALSE book, blob area: %d, blob perimeter: %d\r\n"% (frame_number, area1, perimeter1))
                    cv2.drawContours(frame, contours, j, [0, 0, 255], -1)
                else:
                    x,y,w,h = cv2.boundingRect(cnt)
                    rect_area = w*h
                    area2 = cv2.contourArea(cnt)
                    perimeter2 = cv2.arcLength(cnt, True)
                    extent = float(area2)/rect_area
                    #print(extent)
                    if (extent > 0.7):
                        file.write("frame %d, detected REAL book, blob area: %d, blob perimeter: %d, blob extent: %f\r\n"% (frame_number, area2, perimeter2, extent))
                        cv2.drawContours(frame, contours, j,[0, 255, 0], -1)

        cv2.imshow('contours', frame)
        time.sleep(0.02)

        if (cond==True and hist[255] > 1.0999*prevhist) :
            bg = selective_background_update(bg1, gray, bg, 0.2, closing3)
        elif (cond == True and (closing==prevclos).all==True):
            diff=prevclos-closing
            cv2.imshow('diff', diff)
            bg = selective_background_update(bg1, gray, bg, 0.3, diff)
        
        prevhist=hist[255]
        prevclos=closing
        cond = True
        frame_number += 1
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr, frame,b)
