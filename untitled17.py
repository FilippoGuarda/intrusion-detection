#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:24:41 2021

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
def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def hist_match(original, specified):
 
    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()
 
    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)
 
    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
 
    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')
 
    return b[bin_idx].reshape(oldshape)


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


def pfm(hist):
    total_pixel = np.sum(hist)
    pfm = []
    for i in range(256):
        pfm_i = np.sum(hist[:i]) / total_pixel
        pfm.append(pfm_i)
    return np.asarray(pfm)


#Rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Defining a variable interpolation for mean or median functions
interpolation = np.median  # or np.mean

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


def background_update(bg1,bg, prev_bg, alfa):
    bg1 = (1 - alfa) * prev_bg + alfa * bg
    #bg1=cv2.accumulateWeighted(bg, prev_bg, 0.05)
    #bg1=cv2.GaussianBlur(bg1, (5, 5), 0)
    return bg1

def skip_background(contours, frame, mask, shift1, shift2, index, thresh):
    # ignore contours that are part of the background

    # take two shifted contours, add them and mask using original contours to obtain internal contour
    #print(index)
    cv2.drawContours(shift1, contours, index, 255, -1, offset=(0, 0))
    shift1=cv2.erode(shift1,kernel,iterations=4)
    #cv2.imshow('internal',shift1)
    cv2.drawContours(shift2, contours, index, 255, 10, offset=(0, 0))
    #shift2=cv2.dilate(shift2, kernel, iterations=5)
    shift2=shift2-mask
    #shift2=cv2.erode(shift2,kernel,iterations=4)
    #cv2.imshow('external', shift2)
    internal_contour =(frame[shift1 > 0])
    hist = cv2.calcHist([internal_contour], [0], None, [256], [0, 256])
    external_contour =(frame[shift2 > 0])
    hist1 = cv2.calcHist([external_contour], [0], None, [256], [0, 256])
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

def sobel(img):
    dst1 = cv2.Sobel(img, -1, 1, 0, 3)
    dst2 = cv2.Sobel(img, -1, 0, 1, 3)
    sob = np.maximum(dst1, dst2)
    return sob

k_size = 5
mean_kernel = np.ones([k_size,k_size])/(k_size**2)

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
    bg_inter1=[]
    #prevhist2 = 0
    cond = False
    bg7=bg.astype(np.uint8)
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
        #gray1=linear_stretching(np.copy(gray), 255,170)
        #gray2=cv2.bitwise_not(gray1.astype(np.uint8))
        gray4 = 255-gray1.astype(np.uint8)
        cv2.imshow('gray4', gray4)
        gray3 = np.copy(gray)
        gray3[gray4<255]=np.asarray(0)
        #bg=linear_stretching(np.copy(bg), 255,170)
        #bg2=bg.astype(np.uint8)
        #bg2[gray4<255]=np.asarray(0)
        cv2.imshow('gray34', gray3)
        #cv2.imshow('gray2', bg.astype(np.uint8))
        cv2.imshow('bg', bg)
        #cv2.imshow('gray23', bg2)
        #Compute background suptraction
        mask = (distance(gray, bg) > threshold)

        mask = mask.astype(np.uint8) * 255
        #mask= fgbg.apply(gray)
        cv2.imshow('mask', mask)
        blur=cv2.GaussianBlur(mask,(5,5),0)
        cv2.imshow('Blur', blur)
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
        #cv2.imshow('thresh', thresh)
        #im_out = thresh | im_floodfill_inv
        #cv2.imshow('combine', im_out)
        # try with opening and closing of the binary image
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 3)
        #cv2.imshow('opening', opening)
        dilated = cv2.dilate(opening, None, iterations=12)
        dilated2 = cv2.bitwise_not(dilated)
        #clos = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 3)
        #cv2.imshow("clos", dilated)
        closing=dilated

        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations = 5)
        #cv2.imshow('closing', closing)
        # dilated = cv2.dilate(opening, None, iterations=2)
        # cv2.imshow('dilated', dilated)
        edges = gray.astype(np.uint8)
        mask[np.logical_not(closing)]=np.asarray(0)
        blur[np.logical_not(closing)]=np.asarray(0)
        mask6 =  (distance(gray, bg) > 17)
        mask6 = mask6.astype(np.uint8) * 255
        mask6[np.logical_not(closing)]=np.asarray(0)
        blur6=cv2.GaussianBlur(mask6,(5,5),3)
        ret6,thresh6 = cv2.threshold(blur6,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('mask6', mask6)
        #cv2.imshow('blur6', blur6)
        #cv2.imshow('thresh6', thresh6)
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
        opening2 = cv2.morphologyEx(thresh6, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 2)
        closing2 = cv2.morphologyEx(opening2, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 2)
        #
        #cv2.imshow('e1', edges2)
        #cv2.imshow('e2', opening2)
        #cv2.imshow('e22', closing2)
        f =gray.astype(np.uint8)
        f[np.logical_not(closing2)]= np.asarray(0)
        #cv2.imshow('e3', f)
        mask1 = cv2.copyMakeBorder(edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        #cv2.imshow('a0', mask1)
        final2 = np.copy(opening)
        cv2.floodFill(final2, mask1, (0, 0), 255);
        im_floodfill_inv = cv2.bitwise_not(final2)
        im_floodfill_inv = cv2.bitwise_xor(im_floodfill_inv, dilated)
        #cv2.imshow('a2', im_floodfill_inv)
        out = closing2
        im=gray.copy()
       # im[np.logical_not(closing2)] = np.asarray(0)
        #cv2.imshow('im', im)
        im2=gray.copy()
        closing3=255-closing
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
       

        hist, bins = np.histogram(mask.flatten(), 256, [0, 256])
        #update background when ligth changes
        #if (cond==True and hist[255] > 1.1*prevhist) :
        #    bg_prev = bg
        #    if cond2==False:
        #        bg = background_update(bg1, gray, bg, 0.1)
        #    elif (cond2==True and hist[255] > 1.1*prevhist > 1.1*prevhist2):
        #        bg = b_up(bg2, gray,bg,bg_prev, 0.05)
        #if (cond2==True):
        #    prevhist2=prevhist
        if (cond==True and hist[255] > 1.1*prevhist) :
            bg9 =[] 
            bg2=bg.astype(np.uint8)
            bg3=bg.astype(np.uint8)
            bg3[np.logical_not(closing3)]=np.asarray(0)
            bg4=bg.astype(np.uint8)
            bg4[np.logical_not(closing)]=np.asarray(0)
            hist1 = cv2.calcHist([bg2], [0], closing3, [256], [0, 256])
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            #print(hist1)
            hist2 = cv2.calcHist([im], [0], closing3, [256], [0, 256])
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            coeff=cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            #l6= linear_stretching(bg,255*coeff, 0)
            #cv2.imshow('bg6', l6.astype(np.uint8))
            print("c", coeff)
            l3= bg2*coeff
            l4= im2+bg4
            #sigma=1.5
            #l5 = cv2.filter2D(l4.astype(np.uint8),-1,mean_kernel)
            #l6 = cv2.GaussianBlur(l4.astype(np.uint8), (k_size,k_size) , sigma)
            #a = hist_match(bg7,l4)
            #cv2.imshow('bg6', l4.astype(np.uint8))
            #cv2.imshow('bg5', a)
            #cv2.imshow('bg7', bg7)
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
            bg = background_update(bg1, l4, bg, 0.2)
            print('change_updated')
            
        elif (cond==True and hist[255] < 0.6*prevhist):
            bg = background_update(bg1, gray, bg, 0.6)
            print('change_updated2')
        prevhist=hist[255]
 
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
        if (blob_count<1):
            bg = background_update(bg1, gray, bg, 0.2)
            print('change_updated2')
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
        cond = True
       # prevfr=gray3-im
        if cv2.waitKey(1) == ord('q'):
            break
    print("Released Video Resource")
    cap.release()
    cv2.destroyAllWindows()


change_detection('1.avi', bg, thr, frame,b)
