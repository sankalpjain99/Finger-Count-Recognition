
"""
Created on Thu Sep 12 11:45:44 2019
@author: Sankalp99
"""

import cv2
import numpy as np
import math
 
capture = cv2.VideoCapture(0)
while(capture.isOpened()):
    ret,frame = capture.read()
    frame = cv2.flip(frame,1)
     
    #Get Hand Data from given rectangle
    cv2.rectangle(frame,(100,100),(300,300),(0,255,5),0)
    crop_img = frame[100:300,100:300]
    
    #Cleaning and Thresholding
    blur = cv2.GaussianBlur(crop_img,(5,5),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY) 
    kernel = np.ones((5,5))
    dilation = cv2.dilate(gray,kernel,iterations=1)
    erosion = cv2.erode(dilation,kernel,iterations=1)
    filtered = cv2.GaussianBlur(erosion,(3,3),0)
    ret,thresh = cv2.threshold(filtered,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
     
    cv2.imshow("Threshold",thresh)
     
    contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
     
    contour = max(contours,key = lambda x:cv2.contourArea(x))
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
     
    hull = cv2.convexHull(contour)
     
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing,[contour],-1,(0,255,0),0)
    cv2.drawContours(drawing,[hull],-1,(0,0,255),0)
     
    hull = cv2.convexHull(contour,returnPoints=False)
    defects = cv2.convexityDefects(contour,hull)
     
    count=0
     
     #COSING FORMULA
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]

        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c)) *180) / 3.14 
        
        if angle <= 90:
            count += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)

        cv2.line(crop_img,start, end, [0,255,0], 2)
        
    if count==0:
        cv2.putText(frame,"ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    elif count == 1:
        cv2.putText(frame,"TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    elif count == 2:
        cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
    elif count == 3:
        cv2.putText(frame,"FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    elif count == 4:
        cv2.putText(frame,"FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
    else:
        pass
        
    
    cv2.imshow('Gesture', frame)
    all_img = np.hstack((drawing,crop_img))
    cv2.imshow('Contours', all_img)
    if cv2.waitKey(1) == ord('q'):
        break
             
capture.release()
cv2.destroyAllWindows()     
         
         
         
         
         
         
         
         
         
         
         
         
         