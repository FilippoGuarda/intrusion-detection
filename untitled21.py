#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:09:40 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 19:28:37 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:25:31 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:28:54 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 18:46:41 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:33:30 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:41:19 2021

@author: sora
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 00:15:15 2021

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

#def NCC(frame,bg):
    #h,w=frame.shape[:2]
    #frame=frame.astype(float)
    #bg=bg.astype(float)
    #y=h*w
    #print(y)
    #s = 2
    #h2 = int(h/s)
    #w2 = int(w/s)
    #res = np.zeros((h, w), np.uint8)
    #print(np.multiply(bg[1,1], frame[1,1]), bg[1,1], frame[1,1], np.std(frame), np.linalg.norm(frame,2), np.linalg.norm(frame))
    #print('h', h)
    #print('w', w)
    #mean1 = np.mean(frame)
    #mean2 = np.mean(bg)
    #res=np.zeros((h, w), np.uint8)
    #print('cc', mean1, mean2)
   # newf = frame - mean1
    #newb = bg - mean2
    #print('c', mean1, mean2)
    #norm = np.linalg.norm(frame)
    #norm2 = np.linalg.norm(bg)
    #print('cc', norm, norm2)
    #for k in range(2):
        #for k2 in range(2):
            #f=frame[(k*120):(k+1)*120,(k2*160):(k2+1)*160]
            #l = cv2.matchTemplate(bg,f,cv2.TM_CCORR_NORMED)
            #cv2.imshow('cc2',  l.astype(np.uint8))
            #res[(k*120):(k+1)*120,(k2*160):(k2+1)*160]=l[(k*120):(k+1)*120,(k2*160):(k2+1)*160]
            
    #f=frame[0:120,0:160]
    #cv2.imshow('cc14',  f.astype(np.uint8))
    #f1=frame[120:240,0:160]
    #cv2.imshow('cc15',  f1.astype(np.uint8))
    #f2=frame[120:240,160:320]
    #cv2.imshow('cc16',  f2.astype(np.uint8))
    #f3=frame[0:120,160:320]
    #cv2.imshow('cc17',  f3.astype(np.uint8))
    #res = cv2.matchTemplate(bg,frame[1:3,1:3],cv2.TM_CCORR_NORMED)
    #l = cv2.matchTemplate(bg,f,cv2.TM_CCORR_NORMED)
    #l1 = cv2.matchTemplate(bg,f1,cv2.TM_CCORR_NORMED)
    #cv2.imshow('cc2',  l)
    #l2 = cv2.matchTemplate(bg,f2,cv2.TM_CCORR_NORMED)
    #cv2.imshow('cc3',  l1)
    #l3 = cv2.matchTemplate(bg,f3,cv2.TM_CCORR_NORMED)
    #cv2.imshow('cc4',  l2)
    #cv2.imshow('cc5',  l3)
    #res[0:120,0:160]=l[0:120,0:160]
    #res[120:240,0:160]=l1[120:240,0:160]
    #res[120:240,160:320]=l2[120:240,160:320]
    #res[0:120,160:320]=l3[0:120,160:320]

    #res = np.dot(f1,bg)/(np.std(f)*np.std(bg))
    #for k in range(1,6):
    #    for k2 in range(1,6):
    #        b1=np.zeros((h, w), np.uint8)
    #        f=frame
    #        b=bg[((k-1)*48+1):(k)*48,((k2-1)*64 + 1):(k2)*64]
    #        b1[((k-1)*48+1):(k)*48,((k2-1)*64 + 1):(k2)*64] = b
    #        cv2.imshow('cc',  b1.astype(np.uint8))
    #        l = (f*b1)/(np.std(f)*np.std(b))
    #        cv2.imshow('cc2',  l.astype(np.uint8))
    #        res1.append(l)
    #l= np.multiply(newf,newb)/(np.std(frame)*np.std(bg))
   # print(l[140,140])
    #l = l > 0.9
    #cv2.imshow('cc',  255 - l.astype(np.uint8)*255)
    
    #for k in range(2):
    #    for k2 in range(2):
    #        f=frame[(k*h2):(k+1)*h2,(k2*w2):(k2+1)*w2]
    #        f=f.astype(float)
    #        bg=bg.astype(float)
    #        b=bg[(k*h2):(k+1)*h2,(k2*w2):(k2+1)*w2]
    #        
    #        l1 = np.multiply(f,b)/(np.linalg.norm(f,2)*np.linalg.norm(b,2))
    #        print(l1)
    #        l1 = l1 
            #cv2.imshow('cc2', l1.astype(np.uint8)*255)
    #        res[(k*h2):(k+1)*h2,(k2*w2):(k2+1)*w2]=l1
            
            
    #for k in range(24):
    #    for k2 in range(32):
    #        f=frame[(k*24):(k+1)*24,(k2*32):(k2+1)*32]
    #        b=bg[(k*24):(k+1)*24,(k2*32):(k2+1)*32]
    #        l1 = (np.multiply(b,f)/(np.std(b)*np.std(f)))
            #print(np.linalg.norm(f), np.std(b), l1[1,1])
            #/(np.std(f)*np.std(b))
            #cv2.imshow('cc2',  l1.astype(np.uint8))
    #        res[(k*24):(k+1)*24,(k2*32):(k2+1)*32]=l1
    #res = (frame)/np.std(frame)
    #/(np.linalg.norm(newf)*np.linalg.norm(newb))
    #for k in range(10):
    #    for i in range(24):
    #        for j in range(32):
    #            l = cv2.matchTemplate(bg[k:(k-1)*24,k:(k-1)*32],frame[i,j],cv2.TM_CCORR_NORMED)
    #            res[k:(k-1)*24,k:(k-1)*32] = l+0
    #l= np.corrcoef(frame, bg)
    #for k in range(1,6):
    #    for k2 in range(1,6):
                #l= cv2.matchTemplate(bg,frame[((k-1)*48+1):(k)*48,((k2-1)*64 + 1):(k2)*64],cv2.TM_CCORR_NORMED)
                #res[((k-1)*48+1):(k)*48,((k2-1)*64 + 1):(k2)*64] = l[((k-1)*48+1):(k)*48,((k2-1)*64 + 1):(k2)*64]
                #res.append(l)
    #res_inter = np.stack(res, axis=0)
    #res_inter = interpolation(res_inter, axis=0)
    #res = cv2.matchTemplate(bg,frame[24*9:240,32*9:320],cv2.TM_CCORR_NORMED)
    #x=np.linalg.norm(frame)
    #y=np.linalg.norm(bg)
    #res = (np.convolve(frame,bg,'valid')) / (x*y)
    #res = 255- res
    #res = l
    #cv2.imshow('cc1', res.astype(np.uint8)*255)
    #return res

def L1(img1, img2):
    diff = np.abs(img1 - img2)
    print(len(img1.shape))
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        diff = np.sum(diff, axis=-1)
    return diff


def L2(img1, img2):
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

def linear_stretching(img, max_value, min_value):
    img[img<min_value] = min_value
    img[img>max_value] = max_value
    linear_stretched_img = 255./(max_value-min_value)*(img-min_value)
    return linear_stretched_img

def exponential_operator(img, r):
    exp_img = ((img/255)**r) *255
    return exp_img

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



#Rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean


def selective_background_initialization(bg, n, cap,count):
    previous_frames = []
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret or not frame is None:
            if count < 2 :
                # initialize previous frames properly
                previous_frames.append(frame.astype(float))
            else:
                frame = frame.astype(float)
                threeframedifference(frame, previous_frames, distance,5)
                cv2.imshow('gf', frame)
                # update previous frames
                previous_frames.pop(0)
                previous_frames.append(frame)
            count += 1
        else:
            break
    cap.release()
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    cv2.destroyAllWindows()
    return [bg_inter, count]

def background_initialization(bg, n, cap, count):
    n=n*2
    while cap.isOpened() and count < n:
        ret, frame = cap.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if ret or not frame is None:
            # Release the Video if ret is false
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hist,bins = np.histogram(frame.flatten(),256,[0,256])
            # eq_op = pfm(hist)*255
            # frame = eq_op[frame]
            # frame
            #if count == 0:
            #    bg.append(frame)
            #else:
            #    bg.append(alfa * frame)
                #bg[count] = (1 - alfa) * bg[count - 1] + bg[count]
            if (count % 2 != 0): 
                #frame = cv2.GaussianBlur(frame, (5, 5), 0)
                bg.append(frame)
            count += 1
            # print(count)
        else:
            break
    cap.release()
    b = bg.copy()
    bg_inter = np.stack(bg, axis=0)
    bg_inter = interpolation(bg_inter, axis=0)
    #bg_inter = linear_stretching(np.copy(bg_inter), 255,250)
    #bg_inter = cv2.GaussianBlur(bg_inter, (5, 5), 0)
    cv2.destroyAllWindows()
    return [b, bg_inter, count]

def selective_background_update(bg1, frame, prev_bg, alfa,closing):
    frame[np.logical_not(closing)] = np.asarray(0)
    #cv2.imshow('g1', frame)
    
    bg2 = np.copy(prev_bg)
    bg3 = np.copy(prev_bg)
    prev_bg = prev_bg.astype(np.uint8)
    prev_bg[np.logical_not(closing)] = np.asarray(0)
    #cv2.imshow('g2', prev_bg)
    bg3[closing==255] = np.asarray(0)
    bg2= (1 - alfa) * prev_bg + alfa * frame +bg3
    bg1 = np.copy(bg2)
    #cv2.imshow('g4', bg2.astype(np.uint8))
    return bg1


def background_update(bg1,bg, prev_bg, alfa):
    bg1 = (1 - alfa) * prev_bg + alfa * bg
    #bg1=cv2.accumulateWeighted(bg, prev_bg, 0.05)
    #bg1=cv2.GaussianBlur(bg1, (5, 5), 0)
    return bg1

def skip_background(contours, frame, final, shift1, shift2, index, thresh):
    # ignore contours that are part of the background

    # take two shifted contours, add them and mask using original contours to obtain internal contour
    #print(index)
    cv2.drawContours(shift1, contours, index, 255, 10, offset=(0, 0))
    shift1=cv2.erode(shift1,kernel,iterations=4)
    #cv2.imshow('internal',shift1)
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(0, 0))
    shift2=cv2.dilate(shift2, kernel, iterations=5)
    shift2=shift2-final
    shift2=cv2.erode(shift2,kernel,iterations=4)
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

#def b_up(bg2,new,bg,prev,alfa):
    #new2=new
    #bg2.append(new)
    #bg2.append(bg)
    #bg2.append(prev)
    #new1=interpolation(bg2)
    #new2 = (1 - alfa) * new1 + alfa * new2
    #return new2
###Define change detection parameters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thr = 30
distance = L2
bg = []
b=[]
bg1=[]
bg2=[]
frame=[]
N_frames = 30 # then refresh
#denoising_kernel = np.array([
#            [1,2,1],
#            [2,4,2],
#            [1,2,1]])/16

k_size = 5
mean_kernel = np.ones([k_size,k_size])/(k_size**2)

# blob detector parameters

cap = cv2.VideoCapture('1.avi')
count = 0

# computation of the background
[b,bg, count] = background_initialization(bg, N_frames, cap, count)


#fgbg = cv2.createBackgroundSubtractorKNN(1,10,False)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def change_detection(video_path, bg, threshold,frame,b):
    # previous_frames = []
    cap = cv2.VideoCapture(video_path)
    prevhist = 0
    #bg_inter1=[]
    #prevhist2 = 0
    cond = False
    #bg7=bg.astype(np.uint8)
    #cond2 = False
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
        gray1=linear_stretching(np.copy(gray), 255,170)
        #gray2=cv2.bitwise_not(gray1.astype(np.uint8))
        gray4 = 255-gray1.astype(np.uint8)
        #cv2.imshow('gray4', gray4)
        gray3 = np.copy(gray)
        gray3[gray4<255]=np.asarray(0)
        #bg=linear_stretching(np.copy(bg), 255,170)
        #bg2=bg.astype(np.uint8)
        #bg2[gray4<255]=np.asarray(0)
        #cv2.imshow('gray34', gray3)
        #cv2.imshow('gray2', bg.astype(np.uint8))
        cv2.imshow('bg', bg)
        #cv2.imshow('gray23', bg2)
        #Compute background suptraction
        mask = (distance(gray, bg) > 0.5)
        #m  = NCC(gray,bg.astype(np.uint8))
        #mask7 = m
        #mask7 = (mask7.astype(np.uint8) * 255)
        mask = mask.astype(np.uint8) * 255
        #mask= fgbg.apply(gray)
        cv2.imshow('mask', mask)
        #cv2.imshow('mask7', mask7.astype(np.uint8))
        blur=cv2.GaussianBlur(mask,(5,5),0)
        #cv2.imshow('Blur', blur)
        #blur2 = cv2.filter2D(blur,-1,denoising_kernel)
        #blur2=cv2.fastNlMeansDenoising(blur)
        #cv2.imshow('Blur2', blur2)
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #h, w = gray.shape[:2]
        #mask1 = np.zeros((h+2, w+2), np.uint8)
        #thresh2=np.copy(thresh)
        #cv2.floodFill(thresh2, mask1, (0,0), 255);
        #cv2.imshow('floodfill1', thresh2)
        # Invert floodfilled image
        #im_floodfill_inv = cv2.bitwise_not(thresh2)
    	# Combine the two images to get the foreground.
        #cv2.imshow('floodfill', im_floodfill_inv)
        cv2.imshow('thresh', thresh)
        #im_out = thresh | im_floodfill_inv
        #cv2.imshow('combine', im_out)
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        cv2.imshow('opening', opening)
        dilated = cv2.dilate(opening, None, iterations=10)
       # dilated2 = cv2.bitwise_not(dilated)
        #clos = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 3)
        cv2.imshow("clos", dilated)
        closing=dilated
        inv_closing=255-closing
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 5)
        cv2.imshow('inv_closing', inv_closing)
        # dilated = cv2.dilate(opening, None, iterations=2)
        # cv2.imshow('dilated', dilated)
        #edges = gray.astype(np.uint8)
        mask[np.logical_not(closing)]=np.asarray(0)
        blur[np.logical_not(closing)]=np.asarray(0)
        mask6 =  (distance(gray, bg) > 0.26)
        mask6 = mask6.astype(np.uint8) * 255
        mask6[np.logical_not(closing)]=np.asarray(0)
        blur6=cv2.GaussianBlur(mask6,(5,5),3)
        ret6,thresh6 = cv2.threshold(blur6,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('mask6', mask6)
        #cv2.imshow('blur6', blur6)
        cv2.imshow('thresh6', thresh6)
        #edges[np.logical_not(closing)] = np.asarray(0)
        #edges18 =bg.astype(np.uint8)
        #edges18[np.logical_not(closing)] = np.asarray(0)
        #cv2.imshow('e', edges)
        #find edges and use as a mask for floodfill
        #edges = cv2.Canny(edges,0,200)
        
        #edges = auto_canny(edges)
        #edges18 = auto_canny(edges18)
        #cv2.imshow('etrue', edges)
        #cv2.imshow('eback', edges18)
        #edges = sobel(edges)
        #edges3= gray - edges
        #cv2.imshow('etrue3', edges3)
        #edges2 = cv2.bitwise_xor(mask, edges)
        opening2 = cv2.morphologyEx(thresh6, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 2)
        #
        #cv2.imshow('e1', edges2)
        cv2.imshow('e2', opening2)
        cv2.imshow('e22', closing2)
        f =gray.astype(np.uint8)
        f[np.logical_not(closing2)]= np.asarray(0)
        #cv2.imshow('e3', f)
        #mask1 = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        #cv2.imshow('a0', mask1)
        #final2 = np.copy(opening)
        #cv2.floodFill(final2, mask1, (0, 0), 255);
        #im_floodfill_inv = cv2.bitwise_not(final2)
        #im_floodfill_inv = cv2.bitwise_xor(im_floodfill_inv, dilated)
        #cv2.imshow('a2', im_floodfill_inv)
        out = closing2
       # im[np.logical_not(closing2)] = np.asarray(0)
        #cv2.imshow('im', im)
        im2=gray.copy()
        closing4=cv2.dilate(closing2, None, iterations=2)
        cv2.imshow('closing4', closing4)
        closing3=255-closing4
        im2[np.logical_not(closing3)] = np.asarray(0)
        cv2.imshow('im2', im2)
        #cv2.imshow('floodfill', out)
        #out= cv2.morphologyEx(out, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 1)
        #cv2.imshow('floodfillclose', out)
        #h, w = gray.shape[:2]
        #mask1 = np.zeros((h+2, w+2), np.uint8)
        #thresh2=np.copy(closing)
        #cv2.floodFill(thresh2, mask1, (0,0), 255);
        #cv2.imshow('floodfill1', thresh2)
        # Invert floodfilled image
        #im_floodfill_inv = cv2.bitwise_not(thresh2)
    	# Combine the two images to get the foreground.
        #cv2.imshow('floodfill', im_floodfill_inv)
        #cv2.imshow('thresh', closing)
        #im_out = closing | im_floodfill_inv
        #cv2.imshow('combine', im_out)
       

        hist, bins = np.histogram(thresh.flatten(), 256, [0, 256])
        #update background when ligth changes
        #if (cond==True and hist[255] > 1.1*prevhist) :
        #    bg_prev = bg
        #    if cond2==False:
        #        bg = background_update(bg1, gray, bg, 0.1)
        #    elif (cond2==True and hist[255] > 1.1*prevhist > 1.1*prevhist2):
        #        bg = b_up(bg2, gray,bg,bg_prev, 0.05)
        #if (cond2==True):
        #    prevhist2=prevhist
       
 
        #for i in range(len(contours)): 
        # reducing treshold augments detection capability, but more false positives
        #    if skip_background(contours, frame, original_contour, shift1, shift2, i, 20):
        #        continue
        #    else:
        #        object_detector(contours, i, image_external, colored_contours, frame_number)
        
        #keypoints = detector_person.detect(dilated)
        #im_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
        #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #keypoints2 = detector_book.detect(closing)
        #use keypoints to update background
        #frame = cv2.drawKeypoints(gray, keypoints2, np.array([]), (255, 0, 0),
        #                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow('Video', frame)
        #_, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        #edged = cv2.Canny(thresh, 60, 200)
        #cv2.imshow('Video', edged)
        
        _, contours, hierarchy = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        final = out
        blob_count = len(contours)
        #for c in contours:
        #    hull = cv2.convexHull(c)
        #    cv2.drawContours(frame, [hull], 0, (0, 255, 0),2)  
            #cv2.drawContours(frame, c, 0, (0, 255, 0), 2) 
            #param = cv2.arcLength(c, True)
            # Approximate what type of shape this is
            #approx = cv2.approxPolyDP(c, 0.01 * param, True)
        #cv2.imshow("hull", frame)
        #for i, cnt in enumerate(contours):
            # if the size of the contour is greater than a threshold
        #    if cv2.contourArea(cnt) < 6000:
        #        continue
            #elif cv2.contourArea(cnt) < 2000:
            #    cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)  # if >0 shows contour
        #    else:
        #        (x, y, w, h) = cv2.boundingRect(cnt)
        #        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                #cv2.drawContours(im_keypoints2, [cnt], 0, (255, 255, 255), 3)

        #cv2.imshow('contours', frame)
        if (hist[255] < 0.3*prevhist):
            bg = selective_background_update(bg1, gray, bg, 0.2, closing3)
            #bg = background_update(bg1, im2, bg, 0.1)
            print('background update')
        #cv2.resizeWindow('contours', 500, 500)
        #image_external = np.zeros(final.shape, np.uint8)
        #colored_contours = np.zeros(frame.shape)
        #original_contour = np.zeros(final.shape, np.uint8)
        shift2 = np.zeros(final.shape, np.uint8)
        shift1 = np.zeros(final.shape, np.uint8)

        for i, cnt in enumerate(contours):
             #person detector
             if cv2.contourArea(cnt)>6000:
                 #draw person in blue
                cv2.drawContours(frame, contours,i,[255, 0, 0], -1)

        for j, cnt in enumerate(contours):
            # object detector
            if cv2.contourArea(contours[j]) < 100 or cv2.contourArea(contours[j]) > 2000:
                continue
            elif skip_background(contours, frame, final , shift1, shift2, j, 0.9) == True:
                #draw false object in red
                cv2.drawContours(frame, contours, j, [0, 0, 255], -1)
            else :
                #draw true objects in green
                cv2.drawContours(frame, contours, j,[0, 255, 0], -1)
        
        cv2.imshow('contours', frame)
        time.sleep(0.02)
       # if (cond==True):
       #     cond2 = True
        if (cond==True and hist[255] > 1.1*prevhist) :
            #bg9 =[] 
           # bg2=bg.astype(np.uint8)
            #bg3=bg.astype(np.uint8)
            #bg3[np.logical_not(closing3)]=np.asarray(0)
            #bg4=bg.astype(np.uint8)
            #bg4[np.logical_not(closing4)]=np.asarray(0)
            #hist1, bins1 = np.histogram(bg3.flatten(), 256, [0, 256])
            #hist2, bins2 = np.histogram(im2.flatten(), 256, [0, 256])
            #alfa=0.2
            #print('back hist', hist1[200:249])
            #print('fr hist', hist2[200:249])
            #hist1 = cv2.calcHist([bg2], [0], closing3, [256], [0, 256])
            #cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            #print(hist1)
            #hist2 = cv2.calcHist([im], [0], closing3, [256], [0, 256])
            #cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            #coeff=cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            #l6= linear_stretching(bg,255*coeff, 0)
            #cv2.imshow('bg6', l6.astype(np.uint8))
            #print("c", coeff)
            #l3= bg2*coeff
            #l4= im2+bg4
            #sigma=1.5
            #l5 = cv2.filter2D(l4.astype(np.uint8),-1,mean_kernel)
            #l6 = cv2.GaussianBlur(l4.astype(np.uint8), (k_size,k_size) , sigma)
            
            #a = hist_match(bg7,l4)
            #cv2.imshow('bg6', l4.astype(np.uint8))
            #cv2.imshow('bg5', a)
            #cv2.imshow('bg7', bg3)
            #l4=cv2.normalize(l4/5, None, 0, 255, cv2.NORM_MINMAX)
            #cv2.imshow('bg3', l3.astype(np.uint8))
            #near_img = cv2.resize(l4,None, fx = 10, fy = 10, interpolation = cv2.INTER_NEAREST)
            #cv2.imshow('bg5', near_img)
            #b.pop(1)
            #b.append(gray)
            #bg_inter1 = np.stack(b, axis=0)            
            #bg_inter1 = np.median(bg_inter1, axis=0)
            #cv2.imshow('bg5', bg_inter1.astype(np.uint8))
            #bg = bg_inter1 
            #cv2.imshow('bg8', bg.astype(np.uint8))
            #bg9.append(l4.astype(np.uint8))
            #bg9.append(bg.astype(np.uint8))
            #bg_inter1 = np.stack(bg9, axis=0)            
            #bg_inter1 = np.median(bg_inter1, axis=0)
            #cv2.imshow('bg8', bg_inter1.astype(np.uint8))
            #bg=bg_inter1
            #bg =l3
            #l = (gray3)-im
            #l2 = (l+bg2)/2
            #print('change_l2', l)
            #cv2.imshow('l2', l2.astype(np.uint8))
            #cv2.imshow('l2', l2.astype(np.uint8))
            #bg = background_update(bg1, gray, bg, 0.05)
            bg = selective_background_update(bg1, gray, bg, 0.2, closing3)
            #bg = background_update(bg1, l4, bg, 0.2, inv_closing)
            #print('change_updated')
            
       # elif (cond==True and hist[255] == 0):
       #     bg = background_update(bg1, gray, bg, 0.2)
            #bg = gray + 0
       #     print('change_updated6')
        prevhist=hist[255]
        #prevhist2=hist[0]
        cond = True
       # prevfr=gray3-im
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr, frame,b)
