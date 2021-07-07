import math
import numpy as np
import cv2
from PIL import Image as im
import numpy.linalg as npla
import dlib
import os
from pathlib import Path 


def transform(point, center, scale, resolution):
    pt = np.array ( [point[0], point[1], 1.0] )
    h = 200.0 * scale
    m = np.eye(3)
    m[0,0] = resolution / h
    m[1,1] = resolution / h
    m[0,2] = resolution * ( -center[0] / h + 0.5 )
    m[1,2] = resolution * ( -center[1] / h + 0.5 )
    m = np.linalg.inv(m)
    return np.matmul (m, pt)[0:2]


def crop(image, center, scale, resolution=256.0):
    ul = transform([1, 1], center, scale, resolution).astype( np.int )
    br = transform([resolution, resolution], center, scale, resolution).astype( np.int )

    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1] ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    
    #print(newImg)
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
    return newImg, [newX, newY, oldX, oldY]


def draw_landmark(x):
    
    background = np.zeros((256,256,3)).astype(np.uint8)
    
    for idx in x:
        w = int(idx[0]),
        h=int(idx[1])
        cv2.circle(background, (w,h), 1, (0,0,255), 1)
        
    return background


class face_detection:
    
    def __init__(self,mode):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        self.mode = mode
    
    def face_detect(self,x):
        # x = x
        rects = self.detector(x)
        try:
            assert rects
            for rect in rects:
                l = rect.left()
                r = rect.right()
                t = rect.top()
                b = rect.bottom()
        except:
            print(f'print rects : {rects}')
        if self.mode =='basic':
            return x[t:b,l:r,::-1]
        elif self.mode == 'whole_face':
            try:
                out = np.array([l,r,t,b])
            except:
                pass
            return out
    
    def landmark_detection(self,x):
    
        img = x
        a= self.detector(img,1)

        if len(a)==0:
            # print('no detection')
            return None
        for face in a:

            landmarks = self.predictor(img, face)
            landmark_list = np.zeros((68,2))

            for i,p in enumerate(landmarks.parts()):
                landmark_list[i] = np.array([p.x, p.y])
        return landmark_list
    
    def run(self, x, box=False):
        
        if self.mode =='basic':
            
            # face_img = self.face_detect(x)
            landmark = self.landmark_detection(x)
            face_img = cv2.resize(x,(256,256),interpolation=cv2.INTER_LINEAR)
            #landmark_img = draw_landmark(landmark)
            
            return face_img, landmark
            
        elif self.mode=='whole_face':
            
            rect = self.face_detect(x)
            x, boxes = self.head_resize(x,rect)
            # landmark = self.landmark_detection(x)
            #landmark_img = draw_landmark(landmark)
            if box:
                return x, boxes
            return x #, landmark
            
        
    def head_resize(self, x,rects):
        real_img = x
        #get x point
        xmax = rects[1];xmin = rects[0]
        ymax = rects[3];ymin = rects[2]

        scale = (xmax - xmin + ymax - ymin) / 240 
        center = np.array( [ (xmin + xmax) / 2.0, (ymin + ymax) / 2.0] )
        centers =[center]
        image =[]

        for c in centers:

            real_re_size, data = crop(real_img,c,scale)

        return real_re_size, data
