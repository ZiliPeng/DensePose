# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################
"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'''
运行脚本：
python notebooks/slimming_points_multi_images.py \
    --cfg configs/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml \
    --output-dir DensePoseData/infer_out/ziliInfer/ \
    --image-ext jpg \
    --wts models/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl \
    DensePoseData/demo_data/test_jpg
'''


from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import json
import operator

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


import numpy
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import pandas as pd

import copy




def contourPointIDs(contourL,c_point,offset):
    if c_point+offset>=contourL:
        pointId=contourL-1
    elif c_point+offset<0:
        pointId=0
    else:
        pointId=c_point+offset

    return pointId


def filteringKP(All_persons,cX,cY,kX,kY,c2k,outer=True,keep=True):

    choosed=True
    if c2k>0:
        rad=c2k
    else:
        rad=int(np.sqrt((kX-cX)*(kX-cX)+(kY-cY)*(kY-cY)))
    
    if (outer==False) and (keep==True):
        return choosed,rad

    if (outer==False) and (keep==False):
        choosed=False
        return choosed,rad

    if keep==True:
        tempCanvas=np.zeros([All_persons.shape[0],All_persons.shape[1]],dtype=np.uint8)
        
        tempCircle=cv2.circle(tempCanvas, (kX,kY), rad, (255, 0, 0), -1)
        intersection=cv2.bitwise_and(All_persons,tempCircle,mask=tempCanvas)
        # cv2.imwrite("DensePoseData/infer_out/intersection.png",intersection)
        
        S_intersection =np.sum(np.float32(np.greater(intersection,0)))
        print('**************S_intersection*******',S_intersection)
        S_circle=3.14*rad*rad
        print('**************S_circle*************',S_circle)
        S_blank=S_circle-S_intersection
        print('**************S_blank**************',S_blank)

        if S_blank<S_circle/3.0 and outer==True:
            choosed=False



    # radRange=int(np.sqrt(intersection))





    return choosed,rad

#画最外边轮廓的关键点
def drawContourKP(im,contours,cX,cY,All_persons,outer=True,LBRT=0, PartId=2,maxAreaId=0,Front=True,KPfile=None,keeplist=([True,True,True,True,True,True]),radlist=([0,0,0,0,0,0])):
    #LBRT记录取0-左、1-下、2-右、3-上的关键点，保持内外轮廓取点位置保持一致
    # print('the number of contour points:',contours[maxAreaId][:,0,0].size)

    if outer:
        tempD=12.0
    else:
        tempD=10.0

    KPstep=int((contours[maxAreaId][:,0,0].size)/tempD) #四肢长度默认为宽度的两倍；且为直立状态，取与重心坐标水平位置同高的轮廓点；身体左半部分取左边的点，有半部分取右边的点
    # print('step size:',KPstep)
    # print('contour length:',cv2.arcLength(contours[maxAreaId],True))
    # print('center point:',(cX,cY))
    HM = np.argwhere(contours[maxAreaId][:,0,1]==cY)#查找轮廓上与重心同水平位置的点的索引值
    VM=np.argwhere(contours[maxAreaId][:,0,0]==cX) #查找轮廓上与重心同垂直位置的点的索引值
    # print(HM[:,0]) #可能有多个轮廓上的点与重心的水平位置相同
    # print(VM[:,0])

    if(HM[:,0].size<1)|(VM[:,0].size<1):
        print ('can not find any keypoints in the contour!')
        # KPfile.writelines(['!',str(cX),',', str(cY),',R=0','\n']) #异常的点
        # KPfile.writelines(['!',str(cX),',', str(cY),',R=0','\n']) #异常的点
        # KPfile.writelines(['!',str(cX),',', str(cY),',R=0','\n']) #异常的点
        return -1

    # print(contours[maxAreaId][int(HM[0,0]),0,0], contours[maxAreaId][int(HM[0,0]),0,1])
    # print(contours[maxAreaId][int(HM[1,0]),0,0], contours[maxAreaId][int(HM[1,0]),0,1])

    HMmin=contours[maxAreaId][int(HM[0,0]),0,0]
    HMmax=contours[maxAreaId][int(HM[0,0]),0,0]
    VMmin=contours[maxAreaId][int(VM[0,0]),0,1]
    VMmax=contours[maxAreaId][int(VM[0,0]),0,1]

    HMminIdx=int(HM[0,0])
    HMmaxIdx=int(HM[0,0])
    VMminIdx=int(VM[0,0])
    VMmaxIdx=int(VM[0,0])



    for i in range(HM.size):
        print('HM:',(contours[maxAreaId][int(HM[i,0]),0,0], contours[maxAreaId][int(HM[i,0]),0,1]))
        if HMmin>contours[maxAreaId][int(HM[i,0]),0,0]:
            HMmin=contours[maxAreaId][int(HM[i,0]),0,0]
            HMminIdx=int(HM[i,0])
        if HMmax<contours[maxAreaId][int(HM[i,0]),0,0]:
            HMmax=contours[maxAreaId][int(HM[i,0]),0,0]
            HMmaxIdx=int(HM[i,0])
    # print(HMmin,HMmax)
    # print(HMminIdx,HMmaxIdx)


    for i in range(VM.size):
        # print('VM:',(contours[maxAreaId][int(VM[i,0]),0,0], contours[maxAreaId][int(VM[i,0]),0,1]))
        if VMmin>contours[maxAreaId][int(VM[i,0]),0,1]:
            VMmin=contours[maxAreaId][int(VM[i,0]),0,1]
            VMminIdx=int(VM[i,0])
        if VMmax<contours[maxAreaId][int(VM[i,0]),0,1]:
            VMmax=contours[maxAreaId][int(VM[i,0]),0,1]
            VMmaxIdx=int(VM[i,0])

    # print(VMmin,VMmax)
    # print(VMminIdx,VMmaxIdx)

    # LeftX=HMmin 
    # LeftY=cY  #contours[maxAreaId][int(HM[0,0]),0,1])
    # RightX=HMmax
    # RightY=cY #contours[maxAreaId][int(HM[1,0]),0,1]
    # TopX=cX
    # TopY=VMmin
    # BottomX=cX
    # BottomY=VMmax
    LeftX=contours[maxAreaId][HMminIdx,0,0] 
    LeftY=cY  #contours[maxAreaId][int(HM[0,0]),0,1])
    RightX=contours[maxAreaId][HMmaxIdx,0,0]
    RightY=cY #contours[maxAreaId][int(HM[1,0]),0,1]
    TopX=cX
    TopY=contours[maxAreaId][VMminIdx,0,1]
    BottomX=cX
    BottomY=contours[maxAreaId][VMmaxIdx,0,1]


    # print('Left-Bottom-Right-Top:',(LeftX,LeftY),(BottomX,BottomY),(RightX,RightY),(TopX,TopY))

    LeftBody=False
    RightBody=False
    if (PartId ==16) or (PartId ==20) or (PartId == 9) or (PartId ==13):
        LeftBody=True
        RightBody=False
    if (PartId ==15) or (PartId ==19) or (PartId == 10) or (PartId ==14):
        LeftBody=False
        RightBody=True
        # LBRT+=2
    if PartId==2:
        LeftBody=True
        RightBody=True
    
    if outer==True:
        HVchanged=False
        #默认取重心左右两边的轮廓点
        # if (LeftBody==True):
        #     LBRT=0
        # if (RightBody==True):
        #     LBRT=2
        Short2Center=abs(contours[maxAreaId][HMminIdx,0,0] -cX)
        if abs(contours[maxAreaId][HMmaxIdx,0,0]-cX)<Short2Center:
            Short2Center=abs(contours[maxAreaId][HMmaxIdx,0,0]-cX)
        if (abs(contours[maxAreaId][VMminIdx,0,1]-cY)<Short2Center) or (abs(contours[maxAreaId][VMmaxIdx,0,1]-cY)<Short2Center):
            HMminIdx,HMmaxIdx=VMmaxIdx,VMminIdx     #以重心点为原点的坐标系中，轮廓线与坐标轴的所有交点中距离短的点确定取水平方向的点还是垂直方向的点
            Short2Center=abs(contours[maxAreaId][VMminIdx,0,1]-cY)
            HVchanged=True
            LBRT+=1
        # if abs(contours[maxAreaId][VMmaxIdx,0,1]-cY)<Short2Center:
        #     HMminIdx,HMmaxIdx=VMmaxIdx,VMminIdx     #以重心点为原点的坐标系中，轮廓线与坐标轴的所有交点中距离短的点确定取水平方向的点还是垂直方向的点
        #     Short2Center=abs(contours[maxAreaId][VMmaxIdx,0,1]-cY)
        #     HVchanged=True
        #     LBRT+=1            
        
        

        
        

        # if contours[0][int(HM[0,0]),0,0] < contours[0][int(HM[1,0]),0,0]: #中心点同水平位置的左边点
        #     cv2.circle(im, (contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]), 2, (255, 0, 0), -1)
        #     cv2.circle(im, (contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
        #     cv2.circle(im, (contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
        #     print((contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]))
        #     print((contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]))
        #     print((contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]))
        # else:
        #     cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]), 2, (255, 0, 0), -1)
        #     cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
        #     cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
        #     print((contours[0][int(HM[1,0])-KPstep,0,0], contours[0][int(HM[1,0])-KPstep,0,1]))
        #     print((contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]))
        #     print((contours[0][int(HM[1,0])+KPstep,0,0], contours[0][int(HM[1,0])+KPstep,0,1]))


        

        if (((PartId != 9) and (PartId != 10)) and ((LeftBody==True) and (HVchanged==True))):
            HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
            LBRT+=2
        # if ((RightBody==True) and (HVchanged==True)):
        #     HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
        # if (((PartId == 9) or (PartId == 10)) and ((RightBody==True) and (HVchanged==True))):
        #     HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
        # if ((LeftBody==True) and (HVchanged==True)):
        #     HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
        if (((PartId == 9) or (PartId == 10)) and (HVchanged==True)):
            HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
            LBRT+=2


        #背面若大于该部分的面积的4/5，需要翻转max与min!!!!
        if (Front== False) and (HVchanged==False):
            HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
            LBRT+=2
    else:
        # if (LeftBody==True) and (LBRT!=0):
        #     LBRT+=0
        # if (RightBody==True) and (LBRT!=2):
        #     LBRT+=2
        # LBRT=LBRT%4
        if LBRT==0:
            HMminIdx,HMmaxIdx=HMminIdx,HMmaxIdx
            Short2Center=abs(contours[maxAreaId][HMminIdx,0,0] -cX)
        if LBRT==1:
            HMminIdx,HMmaxIdx=VMmaxIdx,VMminIdx
            Short2Center=abs(contours[maxAreaId][HMminIdx,0,1] -cY)
        if LBRT==2:
            HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
            Short2Center=abs(contours[maxAreaId][HMmaxIdx,0,0] -cX)
        if LBRT==3:
            HMminIdx,HMmaxIdx=VMminIdx,VMmaxIdx
            Short2Center=abs(contours[maxAreaId][HMminIdx,0,1] -cY)

    # if ((LeftBody==True) and (RightBody==False)):
    #     cv2.circle(im, (contours[0][HMminIdx,0,0], contours[0][HMminIdx,0,1]), 2, (255, 0, 0), -1) #左边关键点
    #     cv2.circle(im, (contours[0][HMminIdx+KPstep,0,0], contours[0][HMminIdx+KPstep,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMminIdx-KPstep,0,0], contours[0][HMminIdx-KPstep,0,1]), 2, (255, 0, 0), -1)
    # if ((RightBody==True) and (LeftBody==False)):
    #     cv2.circle(im, (contours[0][HMmaxIdx,0,0], contours[0][HMmaxIdx,0,1]), 2, (255, 0, 0), -1) #右边关键点
    #     cv2.circle(im, (contours[0][HMmaxIdx+KPstep,0,0], contours[0][HMmaxIdx+KPstep,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMmaxIdx-KPstep,0,0], contours[0][HMmaxIdx-KPstep,0,1]), 2, (255, 0, 0), -1)
    # if ((LeftBody==True) and (RightBody==True)):
    #     cv2.circle(im, (contours[0][HMminIdx,0,0], contours[0][HMminIdx,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMminIdx-int(KPstep*0.8),0,0], contours[0][HMminIdx-int(KPstep*0.8),0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMminIdx+int(KPstep*1.2),0,0], contours[0][HMminIdx+int(KPstep*1.2),0,1]), 2, (255, 0, 0), -1) #左边腰部
    #     cv2.circle(im, (contours[0][HMmaxIdx,0,0], contours[0][HMmaxIdx,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMmaxIdx+int(KPstep*0.8),0,0], contours[0][HMmaxIdx+int(KPstep*0.8),0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMmaxIdx-int(KPstep*1.2),0,0], contours[0][HMmaxIdx-int(KPstep*1.2),0,1]), 2, (255, 0, 0), -1) #右边腰部



    offsetId=0

    
    # keep=([True,True,True])  #此为外轮廓关键点的初始化，需要调整内轮廓关键点的初始化为外轮廓的
    # rad=([0,0,0])
    c2k=0

    
    if ((LeftBody==True) and (RightBody==False)):
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1],c2k,outer,keeplist[0])
        keeplist[0],radlist[0]=kept,radi
        if keeplist[0]==True:
            cv2.circle(im, (contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1]), 2, (255, 0, 0), -1) #左边关键点
            KPfile.writelines([str(contours[maxAreaId][HMminIdx,0,0]),',', str(contours[maxAreaId][HMminIdx,0,1]),',R=',str(radlist[0]),'\n'])
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,KPstep)
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[0]*2/3.0),outer,keeplist[1])
        keeplist[1],radlist[1]=kept,radi
        if keeplist[1]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[1]),'\n']) #左上
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,-KPstep)
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[0]*2/3.0),outer,keeplist[2])
        keeplist[2],radlist[2]=kept,radi
        if keeplist[2]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[2]),'\n']) #左下
        else:
            KPfile.writelines(['Filterd\n'])


    if ((RightBody==True) and (LeftBody==False)):
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1],c2k,outer,keeplist[3])
        keeplist[3],radlist[3]=kept,radi
        if keeplist[3]==True:
            cv2.circle(im, (contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1]), 2, (255, 0, 0), -1) #右边关键点
            KPfile.writelines([str(contours[maxAreaId][HMmaxIdx,0,0]),',', str(contours[maxAreaId][HMmaxIdx,0,1]),',R=',str(radlist[3]),'\n'])
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,KPstep)
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[3]*2/3.0),outer,keeplist[4])
        keeplist[4],radlist[4]=kept,radi
        if keeplist[4]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[4]),'\n']) #右上
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,-KPstep)
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[3]*2/3.0),outer,keeplist[5])
        keeplist[5],radlist[5]=kept,radi
        if keeplist[5]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[5]),'\n']) #右下
        else:
            KPfile.writelines(['Filterd\n'])


    if ((LeftBody==True) and (RightBody==True)):
        c2k=int(0.5*np.sqrt((contours[maxAreaId][HMminIdx,0,0]-cX)*(contours[maxAreaId][HMminIdx,0,0]-cX)+(contours[maxAreaId][HMminIdx,0,1]-cY)*(contours[maxAreaId][HMminIdx,0,1]-cY)))
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1],c2k,outer,keeplist[0])
        keeplist[0],radlist[0]=kept,radi
        if keeplist[0]==True:
            cv2.circle(im, (contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1]), 2, (255, 0, 0), -1) #左边关键点
            KPfile.writelines([str(contours[maxAreaId][HMminIdx,0,0]),',', str(contours[maxAreaId][HMminIdx,0,1]),',R=',str(radlist[0]),'\n'])
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,-int(KPstep*0.75))
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[0]*2/3.0),outer,keeplist[1])
        keeplist[1],radlist[1]=kept,radi
        if keeplist[1]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[1]),'\n'])
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,int(KPstep*1.1)) #左边腰部
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[0]*2/3.0),outer,keeplist[2])
        keeplist[2],radlist[2]=kept,radi
        if keeplist[2]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[2]),'\n'])
        else:
            KPfile.writelines(['Filterd\n'])

        c2k=int(0.5*np.sqrt((contours[maxAreaId][HMmaxIdx,0,0]-cX)*(contours[maxAreaId][HMmaxIdx,0,0]-cX)+(contours[maxAreaId][HMmaxIdx,0,1]-cY)*(contours[maxAreaId][HMmaxIdx,0,1]-cY)))
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1],c2k,outer,keeplist[3])
        keeplist[3],radlist[3]=kept,radi
        if keeplist[3]==True:
            cv2.circle(im, (contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1]), 2, (255, 0, 0), -1) #右边关键点
            KPfile.writelines([str(contours[maxAreaId][HMmaxIdx,0,0]),',', str(contours[maxAreaId][HMmaxIdx,0,1]),',R=',str(radlist[3]),'\n'])
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,int(KPstep*0.75))
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[3]*2/3.0),outer,keeplist[4])
        keeplist[4],radlist[4]=kept,radi
        if keeplist[4]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[4]),'\n'])
        else:
            KPfile.writelines(['Filterd\n'])

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,-int(KPstep*1.1)) #右边腰部
        kept,radi=filteringKP(All_persons,cX,cY,contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1],int(radlist[3]*2/3.0),outer,keeplist[5])
        keeplist[5],radlist[5]=kept,radi
        if keeplist[5]==True:
            cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
            KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),',R=',str(radlist[5]),'\n']) 
        else:
            KPfile.writelines(['Filterd\n'])




        # cv2.circle(im, (contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1]), 2, (255, 0, 0), -1)
        # KPfile.writelines([str(contours[maxAreaId][HMminIdx,0,0]),',', str(contours[maxAreaId][HMminIdx,0,1]),'\n'])

        # offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,-int(KPstep*0.75))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        # KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),'\n'])

        # offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,int(KPstep*1.1))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1) #左边腰部
        # KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),'\n'])

        # cv2.circle(im, (contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1]), 2, (255, 0, 0), -1)
        # KPfile.writelines([str(contours[maxAreaId][HMmaxIdx,0,0]),',', str(contours[maxAreaId][HMmaxIdx,0,1]),'\n'])

        # offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,int(KPstep*0.75))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        # KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),'\n'])

        # offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,-int(KPstep*1.1))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1) #右边腰部
        # KPfile.writelines([str(contours[maxAreaId][offsetId,0,0]),',', str(contours[maxAreaId][offsetId,0,1]),'\n'])

    
    LBRT=LBRT%4
    return Short2Center,LBRT,keeplist,radlist


def computeXY(sX,sY,cX,cY,dPX):
    dPC=np.sqrt((sX-cX)*(sX-cX)+(sY-cY)*(sY-cY))
    
    if dPC<=dPX:
        return dPX,cX,cY
    Xx=int(dPX*(cX-sX)/dPC+sX)
    Xy=int(dPX*(cY-sY)/dPC+sY)
    if Xx<0:
        Xx=0
    if Xy<0:
        Xy=0

    return dPC,Xx,Xy


def drawContourKP2(im,contours,cX,cY,outer=True,PartId=2,maxAreaId=0,Front=True,dPX=10):

    print('the number of contour points:',contours[maxAreaId][:,0,0].size)

    if outer:
        tempD=12.0
    else:
        tempD=10.0

    KPstep=int((contours[maxAreaId][:,0,0].size)/tempD) #四肢长度默认为宽度的两倍；且为直立状态，取与重心坐标水平位置同高的轮廓点；身体左半部分取左边的点，有半部分取右边的点
    print('step size:',KPstep)
    print('contour length:',cv2.arcLength(contours[maxAreaId],True))
    print('center point:',(cX,cY))
    HM = np.argwhere(contours[maxAreaId][:,0,1]==cY)#查找轮廓上与重心同水平位置的点的索引值
    VM=np.argwhere(contours[maxAreaId][:,0,0]==cX) #查找轮廓上与重心同垂直位置的点的索引值
    print(HM[:,0]) #可能有多个轮廓上的点与重心的水平位置相同
    print(VM[:,0])

    if(HM[:,0].size<1)|(VM[:,0].size<1):
        return -1

    # print(contours[maxAreaId][int(HM[0,0]),0,0], contours[maxAreaId][int(HM[0,0]),0,1])
    # print(contours[maxAreaId][int(HM[1,0]),0,0], contours[maxAreaId][int(HM[1,0]),0,1])

    HMmin=contours[maxAreaId][int(HM[0,0]),0,0]
    HMmax=contours[maxAreaId][int(HM[0,0]),0,0]
    VMmin=contours[maxAreaId][int(VM[0,0]),0,1]
    VMmax=contours[maxAreaId][int(VM[0,0]),0,1]

    HMminIdx=int(HM[0,0])
    HMmaxIdx=int(HM[0,0])
    VMminIdx=int(VM[0,0])
    VMmaxIdx=int(VM[0,0])



    for i in range(HM.size):
        print('HM:',(contours[maxAreaId][int(HM[i,0]),0,0], contours[maxAreaId][int(HM[i,0]),0,1]))
        if HMmin>contours[maxAreaId][int(HM[i,0]),0,0]:
            HMmin=contours[maxAreaId][int(HM[i,0]),0,0]
            HMminIdx=int(HM[i,0])
        if HMmax<contours[maxAreaId][int(HM[i,0]),0,0]:
            HMmax=contours[maxAreaId][int(HM[i,0]),0,0]
            HMmaxIdx=int(HM[i,0])
    # print(HMmin,HMmax)
    # print(HMminIdx,HMmaxIdx)


    for i in range(VM.size):
        print('VM:',(contours[maxAreaId][int(VM[i,0]),0,0], contours[maxAreaId][int(VM[i,0]),0,1]))
        if VMmin>contours[maxAreaId][int(VM[i,0]),0,1]:
            VMmin=contours[maxAreaId][int(VM[i,0]),0,1]
            VMminIdx=int(VM[i,0])
        if VMmax<contours[maxAreaId][int(VM[i,0]),0,1]:
            VMmax=contours[maxAreaId][int(VM[i,0]),0,1]
            VMmaxIdx=int(VM[i,0])

    # print(VMmin,VMmax)
    # print(VMminIdx,VMmaxIdx)

    # LeftX=HMmin 
    # LeftY=cY  #contours[maxAreaId][int(HM[0,0]),0,1])
    # RightX=HMmax
    # RightY=cY #contours[maxAreaId][int(HM[1,0]),0,1]
    # TopX=cX
    # TopY=VMmin
    # BottomX=cX
    # BottomY=VMmax
    LeftX=contours[maxAreaId][HMminIdx,0,0] 
    LeftY=cY  #contours[maxAreaId][int(HM[0,0]),0,1])
    RightX=contours[maxAreaId][HMmaxIdx,0,0]
    RightY=cY #contours[maxAreaId][int(HM[1,0]),0,1]
    TopX=cX
    TopY=contours[maxAreaId][VMminIdx,0,1]
    BottomX=cX
    BottomY=contours[maxAreaId][VMmaxIdx,0,1]


    print('Left-Bottom-Right-Top:',(LeftX,LeftY),(BottomX,BottomY),(RightX,RightY),(TopX,TopY))
    

    HVchanged=False
    #默认取重心左右两边的轮廓点
    Short2Center=abs(contours[maxAreaId][HMminIdx,0,0] -cX)
    if abs(contours[maxAreaId][HMmaxIdx,0,0]-cX)<Short2Center:
        Short2Center=abs(contours[maxAreaId][HMmaxIdx,0,0]-cX)
    if (abs(contours[maxAreaId][VMminIdx,0,1]-cY)<Short2Center) or abs(contours[maxAreaId][VMmaxIdx,0,1]-cY)<Short2Center:
        HMminIdx,HMmaxIdx=VMmaxIdx,VMminIdx     #以重心点为原点的坐标系中，轮廓线与坐标轴的所有交点中距离短的点确定取水平方向的点还是垂直方向的点
        Short2Center=abs(contours[maxAreaId][VMminIdx,0,1]-cY)
        HVchanged=True
    
    

    
    

    # if contours[0][int(HM[0,0]),0,0] < contours[0][int(HM[1,0]),0,0]: #中心点同水平位置的左边点
    #     cv2.circle(im, (contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
    #     print((contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]))
    #     print((contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]))
    #     print((contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]))
    # else:
    #     cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
    #     print((contours[0][int(HM[1,0])-KPstep,0,0], contours[0][int(HM[1,0])-KPstep,0,1]))
    #     print((contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]))
    #     print((contours[0][int(HM[1,0])+KPstep,0,0], contours[0][int(HM[1,0])+KPstep,0,1]))

    LeftBody=False
    RightBody=False
    if (PartId ==16) or (PartId ==20) or (PartId == 9) or (PartId ==13):
        LeftBody=True
        RightBody=False
    if (PartId ==15) or (PartId ==19) or (PartId == 10) or (PartId ==14):
        LeftBody=False
        RightBody=True
    if PartId==2:
        LeftBody=True
        RightBody=True
    

    if (((PartId != 9) and (PartId != 10)) and ((LeftBody==True) and (HVchanged==True))):
        HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
    # if ((RightBody==True) and (HVchanged==True)):
    #     HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
    # if (((PartId == 9) or (PartId == 10)) and ((RightBody==True) and (HVchanged==True))):
    #     HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
    # if ((LeftBody==True) and (HVchanged==True)):
    #     HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx
    if (((PartId == 9) or (PartId == 10)) and (HVchanged==True)):
        HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx


    #背面若大于该部分的面积的4/5，需要翻转max与min!!!!
    if Front== False:
        HMminIdx,HMmaxIdx=HMmaxIdx,HMminIdx


    # if ((LeftBody==True) and (RightBody==False)):
    #     cv2.circle(im, (contours[0][HMminIdx,0,0], contours[0][HMminIdx,0,1]), 2, (255, 0, 0), -1) #左边关键点
    #     cv2.circle(im, (contours[0][HMminIdx+KPstep,0,0], contours[0][HMminIdx+KPstep,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMminIdx-KPstep,0,0], contours[0][HMminIdx-KPstep,0,1]), 2, (255, 0, 0), -1)
    # if ((RightBody==True) and (LeftBody==False)):
    #     cv2.circle(im, (contours[0][HMmaxIdx,0,0], contours[0][HMmaxIdx,0,1]), 2, (255, 0, 0), -1) #右边关键点
    #     cv2.circle(im, (contours[0][HMmaxIdx+KPstep,0,0], contours[0][HMmaxIdx+KPstep,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMmaxIdx-KPstep,0,0], contours[0][HMmaxIdx-KPstep,0,1]), 2, (255, 0, 0), -1)
    # if ((LeftBody==True) and (RightBody==True)):
    #     cv2.circle(im, (contours[0][HMminIdx,0,0], contours[0][HMminIdx,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMminIdx-int(KPstep*0.8),0,0], contours[0][HMminIdx-int(KPstep*0.8),0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMminIdx+int(KPstep*1.2),0,0], contours[0][HMminIdx+int(KPstep*1.2),0,1]), 2, (255, 0, 0), -1) #左边腰部
    #     cv2.circle(im, (contours[0][HMmaxIdx,0,0], contours[0][HMmaxIdx,0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMmaxIdx+int(KPstep*0.8),0,0], contours[0][HMmaxIdx+int(KPstep*0.8),0,1]), 2, (255, 0, 0), -1)
    #     cv2.circle(im, (contours[0][HMmaxIdx-int(KPstep*1.2),0,0], contours[0][HMmaxIdx-int(KPstep*1.2),0,1]), 2, (255, 0, 0), -1) #右边腰部



    # dPC=np.sqrt((contours[maxAreaId][HMminIdx,0,0]-cX)*(contours[maxAreaId][HMminIdx,0,0]-cX)+(contours[maxAreaId][HMminIdx,0,1]-cY)*(contours[maxAreaId][HMminIdx,0,1]-cY))
    # Xx=int(dPX*(cX-contours[maxAreaId][HMminIdx,0,0])/dPC+contours[maxAreaId][HMminIdx,0,0])
    # Xy=int(dPX*(cY-contours[maxAreaId][HMminIdx,0,1])/dPC+contours[maxAreaId][HMminIdx,0,1])
    # sX=contours[maxAreaId][HMminIdx,0,0]
    # sY=contours[maxAreaId][HMminIdx,0,1]
    # dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
    # print('############################################################################dPC',dPC,Short2Center,(Xx,Xy))




    offsetId=0
    if ((LeftBody==True) and (RightBody==False)):
        # cv2.circle(im, (contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1]), 2, (255, 0, 0), -1) #左边关键点
        sX=contours[maxAreaId][HMminIdx,0,0]
        sY=contours[maxAreaId][HMminIdx,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #左边关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #左边关键内点

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,KPstep)
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点

        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)

        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,-KPstep)
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
    if ((RightBody==True) and (LeftBody==False)):
        # cv2.circle(im, (contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1]), 2, (255, 0, 0), -1) #右边关键点
        sX=contours[maxAreaId][HMmaxIdx,0,0]
        sY=contours[maxAreaId][HMmaxIdx,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #右边关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #右边关键内点


        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,KPstep)
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点



        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,-KPstep)
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点


    if ((LeftBody==True) and (RightBody==True)):
        # cv2.circle(im, (contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1]), 2, (255, 0, 0), -1)
        sX=contours[maxAreaId][HMminIdx,0,0]
        sY=contours[maxAreaId][HMminIdx,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #左边关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #左边关键内点


        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,-int(KPstep*0.75))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点


        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,int(KPstep*1.1))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1) #左边腰部
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #左边腰部
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点        


        # cv2.circle(im, (contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1]), 2, (255, 0, 0), -1)
        sX=contours[maxAreaId][HMmaxIdx,0,0]
        sY=contours[maxAreaId][HMmaxIdx,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #右边关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #右边关键内点


        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,int(KPstep*0.75))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #关键点
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点


        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,-int(KPstep*1.1))
        # cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1) #右边腰部
        sX=contours[maxAreaId][offsetId,0,0]
        sY=contours[maxAreaId][offsetId,0,1]
        dPC,Xx,Xy=computeXY(sX,sY,cX,cY,dPX)
        cv2.circle(im, (sX, sY), 2, (255, 0, 0), -1) #右边腰部
        cv2.circle(im, (Xx, Xy), 2, (255, 0, 0), -1) #关键内点


    return Short2Center




#判断为rePartId身体部分的前视图（true）还是后视图（false）
def judgeFB(IUV,rePartId,thr=0.8):

    Front = True


    # IUVtemp=IUV.reshape(1,-1) #转换成1行
    # print('#####################################',pd.Series(IUVtemp[0]).value_counts())
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',IUVtemp[0].tolist().count(rePartId))
    
    # if rePartId==9 or rePartId==13 or rePartId==10 or rePartId==14:
    #     sizeF=IUVtemp[0].tolist().count(rePartId)
    #     sizeB=IUVtemp[0].tolist().count(rePartId-2)
    # elif rePartId==19 or rePartId==20 or rePartId==15 or rePartId==16:
    #     sizeF=IUVtemp[0].tolist().count(rePartId)
    #     sizeB=IUVtemp[0].tolist().count(rePartId+2)
    # elif rePartId==2:
    #     sizeF=IUVtemp[0].tolist().count(rePartId)
    #     sizeB=IUVtemp[0].tolist().count(rePartId-1)
    # else:
    #     print('无法识别的身体ID代号！')

    if rePartId==9 or rePartId==13 or rePartId==10 or rePartId==14:
        sizeF=IUV.count(rePartId)
        sizeB=IUV.count(rePartId-2)
    elif rePartId==19 or rePartId==20 or rePartId==15 or rePartId==16:
        sizeF=IUV.count(rePartId)
        sizeB=IUV.count(rePartId+2)
    elif rePartId==2:
        sizeF=IUV.count(rePartId)
        sizeB=IUV.count(rePartId-1)
    else:
        print('无法识别的身体ID代号！')

    if sizeB>(sizeF+sizeB)*thr:
        Front = False

    # print('#####################################',rePartId,sizeB,sizeF,Front)


    
    return Front

#20190619新增函数
def findContourKP(im,imName,aPerson,All_persons, IUV, detectedKeypoints, PartIdarray,KPfile):
    #im原图，IUV某个人的IUV,PartIdarray某个人的身体部分ID
    iTempKP=im.copy()
    IUVTempKP=IUV.copy()

    NPKernel = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]

    myKernel=np.uint8(NPKernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))






  

    onePerson = cv2.morphologyEx(aPerson,cv2.MORPH_CLOSE,kernel)
    person_dilated = cv2.dilate(onePerson,kernel,iterations=1)
    _, person_contours, _ = cv2.findContours(person_dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    if len(person_contours)<1:
        print('can not find any contours (before erode)!')
        return
    
    # cntAreas=[]
    maxArea=0
    maxAreaId=0
    for i in range(len(person_contours)):
        # cntAreas.append(cv2.contourArea(contours[i]))
        if cv2.contourArea(person_contours[i])>maxArea:
            maxArea=cv2.contourArea(person_contours[i])
            maxAreaId=i
    
    # max_index, max_cntArea = max(enumerate(cntAreas), key=operator.itemgetter(1))

    # print('the ID of the largest area of a contour:',maxAreaId)
    if maxArea<36:
        print('too small in this area (before erode) to draw inner points!!! ')
        return
    cntPerson=person_contours[maxAreaId]








    print(detectedKeypoints['person_id'])
    aX=np.array(detectedKeypoints['pose_keypoints_x'])
    aX=aX[:,np.newaxis]
    aY=np.array(detectedKeypoints['pose_keypoints_y'])
    aY=aY[:,np.newaxis]
    aL=np.array(detectedKeypoints['pose_keypoints_logit'])
    aL=aL[:,np.newaxis]
    aXY=np.concatenate((aX,aY),axis=1)
    aXYL=np.concatenate((aXY,aL),axis=1)
    print(aXYL[:,2])
    # print(len(aXYL))

    

    IUVtemp=IUV[:,:,0].reshape(1,-1) #转换成1行
    judgeRePartId=2
    Front=judgeFB(IUVtemp[0].tolist(),judgeRePartId,0.25)
    # if Front==False:
    #     continue
    print(Front)

    for index in range(len(PartIdarray)):
        rePartId=PartIdarray[index]
        # print(rePartId)

        # if rePartId==9 or rePartId==13 or rePartId==10 or rePartId==14:
        #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2))
        # elif rePartId==19 or rePartId==20 or rePartId==15 or rePartId==16:
        #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId+2))
        # # elif rePartId==2:
        # #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-1)|(IUV[:,:,0]==3)|(IUV[:,:,0]==4)) #前后身躯+左右手掌
        # else:
        #     print('只通过胳膊、腿找瘦身点')
        #     continue
        
        
        partCanvas = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)


        if rePartId==15 or rePartId==16:
            x,y=np.where((IUV[:,:,0]==1)|(IUV[:,:,0]==2)
                            |(IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId+2)
                            |(IUV[:,:,0]==3)|(IUV[:,:,0]==4)
                            |(IUV[:,:,0]==rePartId+4)|(IUV[:,:,0]==rePartId+6))
        elif rePartId==9 or rePartId==10:
            x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2)
                            |(IUV[:,:,0]==1)|(IUV[:,:,0]==2)
                            |(IUV[:,:,0]==rePartId+2)|(IUV[:,:,0]==rePartId+4))                           
        elif rePartId==13 or rePartId==14:
            x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2)
                            |(IUV[:,:,0]==rePartId-6)|(IUV[:,:,0]==rePartId-4))
        elif rePartId==19 or rePartId==20:
            x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId+2)
                            |(IUV[:,:,0]==rePartId-2)|(IUV[:,:,0]==rePartId-4))

        else:
            print('暂时只处理躯干！')
            continue
        



        partCanvas[x,y]=IUV[x,y]

        personPartI=partCanvas[:,:,0]

        _, person_binary = cv2.threshold(personPartI,0,255,cv2.THRESH_BINARY)  

        person_binary_closed = cv2.morphologyEx(person_binary,cv2.MORPH_CLOSE,kernel)
        dilated = cv2.dilate(person_binary_closed,kernel,iterations=1)
        _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        if len(contours)<1:
            print('can not find any contours (before erode)!')
            continue
        
        # cntAreas=[]
        maxArea=0
        maxAreaId=0
        for i in range(len(contours)):
            # cntAreas.append(cv2.contourArea(contours[i]))
            if cv2.contourArea(contours[i])>maxArea:
                maxArea=cv2.contourArea(contours[i])
                maxAreaId=i
        
        # max_index, max_cntArea = max(enumerate(cntAreas), key=operator.itemgetter(1))

        # print('the ID of the largest area of a contour:',maxAreaId)
        if maxArea<36:
            print('too small in this area (before erode) to draw inner points!!! ')
            continue
        
        
        cntTemp=contours[maxAreaId]

        idts=[0]
        idss=[0]
        # if rePartId==2:
        #     idts=[5,6,11,12]
        if rePartId==15:
            idts=[5]
            idss=[7]
        elif rePartId==16:
            idts=[6]
            idss=[8]
        elif rePartId==9:
            idts=[12]
            idss=[14]
        elif rePartId==10:
            idts=[11]
            idss=[13]
        elif rePartId==13:
            idts=[14]
            idss=[12]
        elif rePartId==14:
            idts=[13]
            idss=[11]
        elif rePartId==19:
            idts=[7]
            idss=[5]
        elif rePartId==20:
            idts=[8]
            idss=[6]
        # if rePartId==2:
        #     idts=[5,6,11,12]
        # elif rePartId==13 or rePartId==14:
        #     idts=[14,13] #左右膝关节
        # elif rePartId==19 or rePartId==20:
        #     idts=[7,8] #左右肘关节


    # keypoints = [
    #     'nose',           #0
    #     'left_eye',       #1
    #     'right_eye',      #2
    #     'left_ear',       #3
    #     'right_ear',      #4
    #     'left_shoulder',  #5
    #     'right_shoulder', #6
    #     'left_elbow',     #7
    #     'right_elbow',    #8
    #     'left_wrist',     #9
    #     'right_wrist',    #10
    #     'left_hip',       #11
    #     'right_hip',      #12
    #     'left_knee',      #13
    #     'right_knee',     #14
    #     'left_ankle',     #15
    #     'right_ankle'     #16
    # ]



        
        # # 轮廓的极点：
        # mostPoints=[]
        # leftmost = tuple(cntTemp[cntTemp[:,:,0].argmin()][0])
        # rightmost = tuple(cntTemp[cntTemp[:,:,0].argmax()][0])
        # topmost = tuple(cntTemp[cntTemp[:,:,1].argmin()][0])
        # bottommost = tuple(cntTemp[cntTemp[:,:,1].argmax()][0])
        # print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",(leftmost,rightmost,topmost,bottommost))
        # mostPoints.append(leftmost)
        # mostPoints.append(rightmost)
        # mostPoints.append(topmost)
        # mostPoints.append(bottommost)
        # for mostPi,mostPoint in enumerate(mostPoints):
        #     cv2.circle(
        #                 iTempKP, mostPoint,
        #                 radius=2, color=(0,0,255), thickness=-1, lineType=cv2.LINE_AA)

        # #cntPerson为此人的整体外轮廓
        # hull = cv2.convexHull(cntTemp,returnPoints = False)
        # defects = cv2.convexityDefects(cntTemp,hull)
        # for i in range(defects.shape[0]):
        #     s,e,f,d = defects[i,0]
        #     start = tuple(cntTemp[s][0])
        #     end = tuple(cntTemp[e][0])
        #     far = tuple(cntTemp[f][0])
        #     # cv2.line(iTempKP,start,end,[0,255,0],2)
        #     # if cv2.pointPolygonTest(cntPerson,start,False)<=0:
        #     #     cv2.circle(iTempKP,start,2,[0,255,0],-1)
        #     # if cv2.pointPolygonTest(cntPerson,end,False)<=0:
        #     #     cv2.circle(iTempKP,end,2,[0,255,0],-1)
        #     if cv2.pointPolygonTest(cntPerson,far,False)<=0:
        #         cv2.circle(iTempKP,far,2,[0,0,255],-1)


        tempR=0
        for _,idt in enumerate(idts):
            print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBb",aXYL[idt,2])

            if aXYL[idt,2]> 1.23:
                if (idt==5 or idt==6) and Front==False:
                    confidShoulder=5
                    if aXYL[6,2]>aXYL[5,2]:
                        confidShoulder=6

                    shortestPY=aXYL[confidShoulder,1]    #水平坐标
                    shortestPX=aXYL[confidShoulder,0]    #垂直坐标
                    tempR=2

                elif (idt==11 or idt==12) and Front==False:
                    confidHip=11
                    if aXYL[12,2]>aXYL[11,2]:
                        confidHip=12

                    shortestPY=aXYL[confidHip,1]    #水平坐标
                    shortestPX=aXYL[confidHip,0]    #垂直坐标
                    tempR=2
                
                elif cv2.pointPolygonTest(cntTemp,(int(aXYL[idt,0]),int(aXYL[idt,1])),False)>=0:
                    # retval=cv2.pointPolygonTest(cntTemp,(int(aXYL[idt,0]),int(aXYL[idt,1])),True)
                    # print(type(retval))
                    # print(len(cntTemp))
                    dist=(cntTemp[:,0,1]-aXYL[idt,1])*(cntTemp[:,0,1]-aXYL[idt,1])+(cntTemp[:,0,0]-aXYL[idt,0])*(cntTemp[:,0,0]-aXYL[idt,0])
                    print(len(dist))

                    min_p2cIndex, min_p2cDist = min(enumerate(dist), key=operator.itemgetter(1))
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    print(min_p2cIndex, np.sqrt(min_p2cDist))

                    shortestPY=cntTemp[min_p2cIndex,0,1]    #水平坐标
                    shortestPX=cntTemp[min_p2cIndex,0,0]    #垂直坐标
                    tempR=0
                    # cv2.circle(
                    #     iTempKP, (int(shortestPX),int(shortestPY)),
                    #     radius=3+tempR, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
                elif cv2.pointPolygonTest(cntTemp,(int(aXYL[idt,0]),int(aXYL[idt,1])),False)<0:
                    print("这个点不在身体该身体部分的区域内：", (int(aXYL[idt,0]),int(aXYL[idt,1])))
                    shortestPY=0    #水平坐标
                    shortestPX=0    #垂直坐标

            else:
                print("未检测到这个关节点：",idt)
                continue
            
            print(int(shortestPX),int(shortestPY))
            if shortestPX!=0 or shortestPY!=0:
                cv2.circle(
                    iTempKP, (int(shortestPX),int(shortestPY)),
                    radius=3+tempR, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
            
             

        

    # keypoints = [
    #     'nose',           #0
    #     'left_eye',       #1
    #     'right_eye',      #2
    #     'left_ear',       #3
    #     'right_ear',      #4
    #     'left_shoulder',  #5
    #     'right_shoulder', #6
    #     'left_elbow',     #7
    #     'right_elbow',    #8
    #     'left_wrist',     #9
    #     'right_wrist',    #10
    #     'left_hip',       #11
    #     'right_hip',      #12
    #     'left_knee',      #13
    #     'right_knee',     #14
    #     'left_ankle',     #15
    #     'right_ankle'     #16
    # ]

    




    for idt in range(len(aXYL)):
        if aXYL[idt,2]> 1.8 and idt!=1 and idt!=2 and idt!=3 and idt!=4:
        # if idt!=1 and idt!=2 and idt!=3 and idt!=4:
            cv2.circle(
                iTempKP, (int(aXYL[idt,0]),int(aXYL[idt,1])),
                radius=1, color=(200,10*(detectedKeypoints['person_id'][0]),100), thickness=-1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(imName+'_iTempKP.png',iTempKP)
    
    return iTempKP



#20190618新增函数
def findOutcontourKP(im,imName,All_persons, IUV, detectedKeypoints, PartIdarray,KPfile):
    #im原图，IUV某个人的IUV,PartIdarray某个人的身体部分ID
    iTempKP=im.copy()
    IUVTempKP=IUV.copy()

    NPKernel = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]

    myKernel=np.uint8(NPKernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11, 11))


    print(detectedKeypoints['person_id'])
    aX=np.array(detectedKeypoints['pose_keypoints_x'])
    aX=aX[:,np.newaxis]
    aY=np.array(detectedKeypoints['pose_keypoints_y'])
    aY=aY[:,np.newaxis]
    aL=np.array(detectedKeypoints['pose_keypoints_logit'])
    aL=aL[:,np.newaxis]
    aXY=np.concatenate((aX,aY),axis=1)
    aXYL=np.concatenate((aXY,aL),axis=1)
    print(aXYL[:,2])
    # print(len(aXYL))

    

    IUVtemp=IUV[:,:,0].reshape(1,-1) #转换成1行
    judgeRePartId=2
    Front=judgeFB(IUVtemp[0].tolist(),judgeRePartId,0.25)
    # if Front==False:
    #     continue
    print(Front)

    for index in range(len(PartIdarray)):
        rePartId=PartIdarray[index]
        # print(rePartId)

        # if rePartId==9 or rePartId==13 or rePartId==10 or rePartId==14:
        #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2))
        # elif rePartId==19 or rePartId==20 or rePartId==15 or rePartId==16:
        #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId+2))
        # elif rePartId==2:
        #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-1)|(IUV[:,:,0]==3)|(IUV[:,:,0]==4)) #前后身躯+左右手掌
        # else:
        #     print('can not recognize the PartID!')
        #     continue
        
        
        partCanvas = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)

        # if rePartId==2:
        #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-1)
        #                     |(IUV[:,:,0]==15)|(IUV[:,:,0]==16)|(IUV[:,:,0]==17)|(IUV[:,:,0]==18)
        #                     |(IUV[:,:,0]==3)|(IUV[:,:,0]==4)
        #                     |(IUV[:,:,0]==7)|(IUV[:,:,0]==8)|(IUV[:,:,0]==9)|(IUV[:,:,0]==10)
        #                     |(IUV[:,:,0]==19)|(IUV[:,:,0]==20)|(IUV[:,:,0]==21)|(IUV[:,:,0]==22))
        if rePartId==15 or rePartId==16:
            x,y=np.where((IUV[:,:,0]==1)|(IUV[:,:,0]==2)
                            |(IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId+2)
                            |(IUV[:,:,0]==3)|(IUV[:,:,0]==4)
                            |(IUV[:,:,0]==rePartId+4)|(IUV[:,:,0]==rePartId+6))
        elif rePartId==9 or rePartId==10:
            x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2)
                            |(IUV[:,:,0]==1)|(IUV[:,:,0]==2)
                            |(IUV[:,:,0]==rePartId+2)|(IUV[:,:,0]==rePartId+4))                           
        elif rePartId==13 or rePartId==14:
            x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2)
                            |(IUV[:,:,0]==rePartId-6)|(IUV[:,:,0]==rePartId-4))
        elif rePartId==19 or rePartId==20:
            x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId+2)
                            |(IUV[:,:,0]==rePartId-2)|(IUV[:,:,0]==rePartId-4))

        else:
            print('暂时只处理躯干！')
            continue
        



        partCanvas[x,y]=IUV[x,y]

        personPartI=partCanvas[:,:,0]

        _, person_binary = cv2.threshold(personPartI,0,255,cv2.THRESH_BINARY)  

        person_binary_closed = cv2.morphologyEx(person_binary,cv2.MORPH_CLOSE,kernel)
        dilated = cv2.dilate(person_binary_closed,kernel,iterations=1)
        _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        if len(contours)<1:
            print('can not find any contours (before erode)!')
            continue
        
        # cntAreas=[]
        maxArea=0
        maxAreaId=0
        for i in range(len(contours)):
            # cntAreas.append(cv2.contourArea(contours[i]))
            if cv2.contourArea(contours[i])>maxArea:
                maxArea=cv2.contourArea(contours[i])
                maxAreaId=i
        
        # max_index, max_cntArea = max(enumerate(cntAreas), key=operator.itemgetter(1))

        # print('the ID of the largest area of a contour:',maxAreaId)
        if maxArea<36:
            print('too small in this area (before erode) to draw inner points!!! ')
            continue
        
        
        cntTemp=contours[maxAreaId]

        idts=[0]
        # if rePartId==2:
        #     idts=[5,6,11,12]
        if rePartId==15:
            idts=[5]
        elif rePartId==16:
            idts=[6]
        elif rePartId==9:
            idts=[12]
        elif rePartId==10:
            idts=[11]
        elif rePartId==13:
            idts=[14]
        elif rePartId==14:
            idts=[13]
        elif rePartId==19:
            idts=[7]
        elif rePartId==20:
            idts=[8]
        # if rePartId==2:
        #     idts=[5,6,11,12]
        # elif rePartId==13 or rePartId==14:
        #     idts=[14,13] #左右膝关节
        # elif rePartId==19 or rePartId==20:
        #     idts=[7,8] #左右肘关节
        

        # 轮廓的极点：
        # leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        # rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        # topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        # bottommost = tuple(cnt[cnt[:,:,1].argmax()][0]

        tempR=0
        for _,idt in enumerate(idts):
            print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBb",aXYL[idt,2])

            if aXYL[idt,2]> 1.23:
                if (idt==5 or idt==6) and Front==False:
                    confidShoulder=5
                    if aXYL[6,2]>aXYL[5,2]:
                        confidShoulder=6

                    shortestPY=aXYL[confidShoulder,1]    #水平坐标
                    shortestPX=aXYL[confidShoulder,0]    #垂直坐标
                    tempR=2

                elif (idt==11 or idt==12) and Front==False:
                    confidHip=11
                    if aXYL[12,2]>aXYL[11,2]:
                        confidHip=12

                    shortestPY=aXYL[confidHip,1]    #水平坐标
                    shortestPX=aXYL[confidHip,0]    #垂直坐标
                    tempR=2
                
                elif cv2.pointPolygonTest(cntTemp,(int(aXYL[idt,0]),int(aXYL[idt,1])),False)>=0:
                    # retval=cv2.pointPolygonTest(cntTemp,(int(aXYL[idt,0]),int(aXYL[idt,1])),True)
                    # print(type(retval))
                    # print(len(cntTemp))
                    dist=(cntTemp[:,0,1]-aXYL[idt,1])*(cntTemp[:,0,1]-aXYL[idt,1])+(cntTemp[:,0,0]-aXYL[idt,0])*(cntTemp[:,0,0]-aXYL[idt,0])
                    print(len(dist))

                    min_p2cIndex, min_p2cDist = min(enumerate(dist), key=operator.itemgetter(1))
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    print(min_p2cIndex, np.sqrt(min_p2cDist))

                    shortestPY=cntTemp[min_p2cIndex,0,1]    #水平坐标
                    shortestPX=cntTemp[min_p2cIndex,0,0]    #垂直坐标
                    tempR=0
                    # cv2.circle(
                    #     iTempKP, (int(shortestPX),int(shortestPY)),
                    #     radius=3+tempR, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
                elif cv2.pointPolygonTest(cntTemp,(int(aXYL[idt,0]),int(aXYL[idt,1])),False)<0:
                    print("这个点不在身体该身体部分的区域内：", (int(aXYL[idt,0]),int(aXYL[idt,1])))
                    shortestPY=0    #水平坐标
                    shortestPX=0    #垂直坐标

            else:
                print("未检测到这个关节点：",idt)
                continue
            
            print(int(shortestPX),int(shortestPY))
            if shortestPX!=0 or shortestPY!=0:
                cv2.circle(
                    iTempKP, (int(shortestPX),int(shortestPY)),
                    radius=3+tempR, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)

        

    # keypoints = [
    #     'nose',
    #     'left_eye',
    #     'right_eye',
    #     'left_ear',
    #     'right_ear',
    #     'left_shoulder',
    #     'right_shoulder',
    #     'left_elbow',
    #     'right_elbow',
    #     'left_wrist',
    #     'right_wrist',
    #     'left_hip',
    #     'right_hip',
    #     'left_knee',
    #     'right_knee',
    #     'left_ankle',
    #     'right_ankle'
    # ]

    




    for idt in range(len(aXYL)):
        if aXYL[idt,2]> 1.8 and idt!=1 and idt!=2 and idt!=3 and idt!=4:
        # if idt!=1 and idt!=2 and idt!=3 and idt!=4:
            cv2.circle(
                iTempKP, (int(aXYL[idt,0]),int(aXYL[idt,1])),
                radius=3, color=(200,10*(detectedKeypoints['person_id'][0]),100), thickness=-1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(imName+'_iTempKP.png',iTempKP)
    
    return iTempKP



def processPart(im,All_persons, IUV, PartIdarray,KPfile):
        xpixels = im.shape[1]
        ypixels = im.shape[0]

        dpi = 300
        scalefactor = 1.0

        xinch = xpixels * scalefactor / dpi
        yinch = ypixels * scalefactor / dpi




        # plt.imshow( im[:,:,::-1] )
        # plt.contour( IUV[:,:,1]/256.,8, linewidths = 1 )
        # plt.contour( IUV[:,:,2]/256.,8, linewidths = 1 )
        # plt.contour( INDS, linewidths = 2 )

        # plt.savefig('../DensePoseData/infer_out/slim/000654_IUVplusIMG.png', dpi=dpi)


        # canvas = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        # plt.imshow( canvas[:,:,::-1] )
        # plt.contour( IUV[:,:,1]/256.,10, linewidths = 1 )
        # plt.contour( IUV[:,:,2]/256.,10, linewidths = 1 )
        # plt.contour( INDS, linewidths = 2 )
        # plt.savefig('../DensePoseData/infer_out/slim/000654_IUVplusBLK.png', dpi=dpi)


        
        xxx=im
       
        
        # PartId=PartIdarray[1]
        # for PartId in PartIdarray:
        for index in range(len(PartIdarray)):
            rePartId=PartIdarray[index]
            # print(rePartId)
            

            

            # Front=judgeFB(IUV[:,:,0],rePartId)
            IUVtemp=IUV[:,:,0].reshape(1,-1) #转换成1行
            # print('#####################################',pd.Series(IUVtemp[0]).value_counts())
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',IUVtemp[0].tolist().count(rePartId))

            Front=judgeFB(IUVtemp[0].tolist(),rePartId)
     

            # # fig = plt.gcf()
            # fig = plt.figure(figsize=(xinch,yinch))
            # # fig.set_size_inches(xinch,yinch)
            # plt.axis('off') 

            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            # plt.margins(0,0)

            canvas2 = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
            # plt.imshow( canvas2[:,:,::-1] )
            
            # for PartID in xrange(1,25):
            #     x,y=np.where(IUV[:,:,0]==PartID)
            #     plt.scatter(y,x,10, np.array(y))

            # x,y=np.where((IUV[:,:,0]==2)|
            #              (IUV[:,:,0]==9)|
            #              (IUV[:,:,0]==10)|
            #              (IUV[:,:,0]==13)|
            #              (IUV[:,:,0]==14)|
            #              (IUV[:,:,0]==23)|
            #              (IUV[:,:,0]==24)|
            #              (IUV[:,:,0]==15)|
            #              (IUV[:,:,0]==16)|
            #              (IUV[:,:,0]==19)|
            #              (IUV[:,:,0]==20)
            #              )
            # x,y=np.where((IUV[:,:,0]!=0))

            #合并前后部分。为防止左右手掌遮盖，也将其与身体躯干合并
            # if rePartId==9 or rePartId==13:
            #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2)|(IUV[:,:,0]==3)) #左腿+左手掌
            # elif rePartId==10 or rePartId==14:
            #     x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2)|(IUV[:,:,0]==4)) #右腿+右手掌
            

            if rePartId==9 or rePartId==13 or rePartId==10 or rePartId==14:
                x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-2))
            elif rePartId==19 or rePartId==20 or rePartId==15 or rePartId==16:
                x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId+2))
            elif rePartId==2:
                x,y=np.where((IUV[:,:,0]==rePartId)|(IUV[:,:,0]==rePartId-1)|(IUV[:,:,0]==3)|(IUV[:,:,0]==4)) #前后身躯+左右手掌
            else:
                print('can not recognize the PartID!')
                continue

            
            

            # plt.scatter(y,x,22, np.array(x))
            # print('canvas2的形状',canvas2.shape)
            canvas2[x,y]=IUV[x,y]
            
            # x,y=np.where((IUV[:,:,0]==rePartId))
            # plt.scatter(y,x,22, np.array(x))
            # # print('canvas2的形状',canvas2.shape)
            # canvas2[x,y]=IUV[x,y]
            
            # saveName='../DensePoseData/infer_out/slim/000654_KeyPointsplusBLK_'+str(rePartId)+'.png'
            # cv2.imwrite(saveName,canvas2)
            # print(saveName)

            # plt.savefig(saveName, dpi=dpi)



            # plt.imshow( canvas[:,:,::-1] )
            # # plt.scatter(INDS[:,0],INDS[:,1],11, np.arange(len(INDS[:,0]))  )
            # # plt.contour( INDS,10, linewidths = 2 )

            # plt.savefig('../DensePoseData/infer_out/slim/000654_ContourKeyPointsplusBLK.png', dpi=dpi)
            # plt.close('all')


            # person0=cv2.imread(saveName,0)
            person0=canvas2[:,:,0]


            _, person_binary = cv2.threshold(person0,0,255,cv2.THRESH_BINARY)  



            NPKernel = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]

            myKernel=np.uint8(NPKernel)


            # _, contours, hierarchy = cv2.findContours(person_binary,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # xxx=cv2.drawContours(im, contours, -1,(0,255,0), 1, cv2.LINE_AA)
            



            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13, 13))
            
            person_binary_closed = cv2.morphologyEx(person_binary,cv2.MORPH_CLOSE,kernel)
            dilated = cv2.dilate(person_binary_closed,kernel,iterations=1)
            _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # xxx=cv2.drawContours(im, contours, -1,(0,255,0), 1, cv2.LINE_AA)
            
            # print(cv2.contourArea(contours))

            # 轮廓的极点：
            # leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            # rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            # topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            # bottommost = tuple(cnt[cnt[:,:,1].argmax()][0]



            if len(contours)<1:
                print('can not find any contours (before erode)!')
                continue
            

            maxArea=0
            maxAreaId=0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i])>maxArea:
                    maxArea=cv2.contourArea(contours[i])
                    maxAreaId=i

            # print('the ID of the largest area of a contour:',maxAreaId)
            if maxArea<36:
                print('too small in this area (before erode) to draw inner points!!! ')
                continue
                

            # xxx=cv2.drawContours(im, contours, maxAreaId,(0,255,0), 1, cv2.LINE_AA)

            KPfile.writelines(['\n','PartID:', str(rePartId),'\n'])

            # compute the center of the contour
            M = cv2.moments(contours[maxAreaId])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(im, (cX, cY), 3, (255, 0, 255), -1)

            """        

                print(contours[0][:,0,0].size)

                KPstep=int((contours[0][:,0,0].size)/12.) #四肢长度默认为宽度的两倍；且为直立状态，取与重心坐标水平位置同高的轮廓点；身体左半部分取左边的点，有半部分取右边的点
                print(KPstep)
                print(cv2.arcLength(contours[0],True))
                # print(cX,cY)
                HM = np.argwhere(contours[0][:,0,1]==cY)
                # print(HM[:,0])
                # print(contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1])
                # print(contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1])
                if contours[0][int(HM[0,0]),0,0] < contours[0][int(HM[1,0]),0,0]: #中心点同水平位置的左边点
                    cv2.circle(im, (contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
                    print((contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]))
                    print((contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]))
                    print((contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]))
                else:
                    cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
                    print((contours[0][int(HM[1,0])-KPstep,0,0], contours[0][int(HM[1,0])-KPstep,0,1]))
                    print((contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]))
                    print((contours[0][int(HM[1,0])+KPstep,0,0], contours[0][int(HM[1,0])+KPstep,0,1]))

            """
            keeplist=([True,True,True,True,True,True])
            radlist=([0,0,0,0,0,0])
            Short2Center,LBRT,keeplist_new,radlist_new=drawContourKP(im,contours,cX,cY,All_persons,True,0,rePartId,maxAreaId=maxAreaId,Front=Front,KPfile=KPfile,keeplist=keeplist,radlist=radlist)
            # drawContourKP2(im,contours,cX,cY,True,rePartId,maxAreaId=maxAreaId,Front=Front,dPX=10)



            
            
            tempArea=cv2.contourArea(contours[maxAreaId])
            if tempArea>300:
                myiter=3
            elif tempArea>200:
                myiter=2
            else:
                myiter=1

            






            eroded2 = cv2.erode(person_binary_closed,myKernel,myiter) #需要根据面积调整迭代次数
            _, contours, hierarchy = cv2.findContours(eroded2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # xxx=cv2.drawContours(im, contours, -1,(0,0,255), 1, cv2.LINE_AA)





            # compute the center of the contour
            # M = cv2.moments(contours[0])
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            # cv2.circle(im, (cX, cY), 3, (255, 0, 255), -1)

            """
                print(contours[0][:,0,0].size)

                KPstep=int((contours[0][:,0,0].size)/10.)
                print(KPstep)
                print(cv2.arcLength(contours[0],True))
                # print(cX,cY)
                HM = np.argwhere(contours[0][:,0,1]==cY)
                # print(HM[:,0])
                # print(contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1])
                # print(contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1])
                if contours[0][int(HM[0,0]),0,0] < contours[0][int(HM[1,0]),0,0]: #中心点同水平位置的左边点
                    cv2.circle(im, (contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
                    print((contours[0][int(HM[0,0])-KPstep,0,0], contours[0][int(HM[0,0])-KPstep,0,1]))
                    print((contours[0][int(HM[0,0]),0,0], contours[0][int(HM[0,0]),0,1]))
                    print((contours[0][int(HM[0,0])+KPstep,0,0], contours[0][int(HM[0,0])+KPstep,0,1]))
                else:
                    cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])+KPstep,0,1]), 2, (255, 0, 0), -1)
                    cv2.circle(im, (contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0])-KPstep,0,1]), 2, (255, 0, 0), -1)
                    print((contours[0][int(HM[1,0])-KPstep,0,0], contours[0][int(HM[1,0])-KPstep,0,1]))
                    print((contours[0][int(HM[1,0]),0,0], contours[0][int(HM[1,0]),0,1]))
                    print((contours[0][int(HM[1,0])+KPstep,0,0], contours[0][int(HM[1,0])+KPstep,0,1]))
            """


            if len(contours)<1:
                
                print('can not find any contours (after erode)!')
                print('All the inner keypoints can be the center point:',(cX,cY))
                if rePartId!=2:
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                elif rePartId==2:
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])

                continue
            

            maxArea=0
            maxAreaId=0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i])>maxArea:
                    maxArea=cv2.contourArea(contours[i])
                    maxAreaId=i
            xxx=cv2.drawContours(im, contours, maxAreaId,(0,0,255), 1, cv2.LINE_AA)
            if maxArea<4:
                print('too small in this area (after erode) to draw inner points!!! ')
                print('All the inner keypoints can be the center point:',(cX,cY))
                if rePartId!=2:
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                elif rePartId==2:
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])
                    KPfile.writelines(['!',str(cX),',', str(cY),'\n'])

                continue


            # 重新计算腐蚀后的重心点坐标compute the center of the contour
            M = cv2.moments(contours[maxAreaId])
            if M["m00"]==0:
                M["m00"]=0.001
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            drawContourKP(im,contours,cX,cY,All_persons,False,LBRT,rePartId,maxAreaId=maxAreaId,Front=Front,KPfile=KPfile,keeplist=keeplist_new,radlist=radlist_new)


            # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            # dilated = cv2.dilate(person_binary,kernel2)
            # _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # xxx=cv2.drawContours(im, contours, -1,(0,0,255), 1, cv2.LINE_AA)



            # for cnt in contours:
            #     hull = cv2.convexHull(cnt)
            #     length = len(hull)
            #     # 如果凸包点集中的点个数大于5
            #     if length > 5:
            #         # 绘制图像凸包的轮廓
            #         for i in range(length):
            #             xxx=cv2.line(im, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 1)
            #             print(tuple(hull[i][0]),'---',tuple(hull[(i+1)%length][0]))



        # cv2.imwrite('../DensePoseData/infer_out/slim/000654_contour.png',xxx)
        return xxx



def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


'''
dp_I: The patch index that indicates which of the 24 surface patches the point is on. 
Patches correspond to the body parts. 
Some body parts are split into 2 patches:
1, 2 = Torso, 
3 = Right Hand, 
4 = Left Hand, 
5 = Left Foot, 
6 = Right Foot, 
7, 9 = Upper Leg Right, 
8, 10 = Upper Leg Left, 
11, 13 = Lower Leg Right, 
12, 14 = Lower Leg Left, 
15, 17 = Upper Arm Left, 
16, 18 = Upper Arm Right, 
19, 21 = Lower Arm Left, 
20, 22 = Lower Arm Right, 
23, 24 = Head
'''
#将自定的身体部分ID转换成IUV中包含前后半的身体部分，再重映射到身体部分ID(1~24)
def pre_process_PartID(procI):
    
    PartIdarray=(16,20,15,19,2,9,13,10,14)

    if procI==1:
        PartIdarray=(2,)  #躯干
    if procI==2:
        PartIdarray=(9,) #左大腿
    if procI==3:
        PartIdarray=(10,) #右大腿
    if procI==4:
        PartIdarray=(13,) #左小腿
    if procI==5:
        PartIdarray=(14,) #右小腿
    if procI==6:
        PartIdarray=(16,) #左上肢
    if procI==7:
        PartIdarray=(15,) #右上肢
    if procI==8:
        PartIdarray=(20,) #左手臂
    if procI==9:
        PartIdarray=(19,) #右手臂
    if procI==0:
        PartIdarray=(16,20,15,19,2,9,13,10,14) #除头部、手掌、脚掌之外的所有部分

    return PartIdarray

def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
            
        IUV_inter0, INDS_inter0,keypoints0=vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            cls_bodys,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )

        # def vis_one_image_opencv(im, boxes, segms=None, keypoints=None, thresh=0.9, kp_thresh=2,
        # show_box=False, dataset=None, show_class=False)
        # def vis_one_image(im, im_name, output_dir, boxes, segms=None, keypoints=None, body_uv=None, thresh=0.9,
        # kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        # ext='pdf')

        # cv2.imwrite(args.output_dir+os.path.basename(im_name).split('.')[0]+'_IUV_inter.png',IUV_inter)
        # cv2.imwrite(args.output_dir+os.path.basename(im_name).split('.')[0]+'_INDS_inter.png',INDS_inter)
        # im  = cv2.imread('../DensePoseData/demo_data/val/000654.jpg')
        # IUV = cv2.imread('../DensePoseData/infer_out/val_res/000654_IUV.png')
        # INDS = cv2.imread('../DensePoseData/infer_out/val_res/000654_INDS.png')
        

        # print('DensePoseData/infer_out/val_results/'+os.path.basename(im_name).split('.')[0]+'_IUV.png')
        # IUV = cv2.imread(args.output_dir+os.path.basename(im_name).split('.')[0]+'_IUV.png')
        # INDS = cv2.imread(args.output_dir+os.path.basename(im_name).split('.')[0]+'_INDS.png')

        IUV=IUV_inter0.copy()
        INDS=INDS_inter0.copy()
        # print(IUV.shape)
        # print(INDS.shape)
        # print(im.shape)

        # print(np.array(keypoints0).shape)
        # print(keypoints0)
        if len(keypoints0)>0:
            test_dict={'file_name': im_name, 'people': keypoints0}
            # print(test_dict)
            # print(type(test_dict))
            json_str = json.dumps(test_dict)
            # print(json_str)
            # print(type(json_str))
            with open(args.output_dir + os.path.basename(im_name).split('.')[0]+'_KP.json', 'w') as json_file:
                    json_file.write(json_str)

        imKP=IUV.copy()
        for i in range(len(keypoints0)):
            print(keypoints0[i]['person_id'])
            aX=np.array(keypoints0[i]['pose_keypoints_x'])
            aX=aX[:,np.newaxis]
            aY=np.array(keypoints0[i]['pose_keypoints_y'])
            aY=aY[:,np.newaxis]
            aL=np.array(keypoints0[i]['pose_keypoints_logit'])
            aL=aL[:,np.newaxis]
            aXY=np.concatenate((aX,aY),axis=1)
            aXYL=np.concatenate((aXY,aL),axis=1)
            print(aXYL[:,2])
            # print(len(aXYL))

            for idt in range(len(aXYL)):
                if aXYL[idt,2]> 1.8 and idt!=1 and idt!=2 and idt!=3 and idt!=4:
                # if idt!=1 and idt!=2 and idt!=3 and idt!=4:
                    cv2.circle(
                        imKP, (int(aXYL[idt,0]),int(aXYL[idt,1])),
                        radius=3, color=(200,10*(keypoints0[i]['person_id'][0]),100), thickness=-1, lineType=cv2.LINE_AA)

        cv2.imwrite(args.output_dir + os.path.basename(im_name).split('.')[0]+'_KP_densepose.jpg',imKP)

        

        if (IUV is None) or (INDS is None):
            print('can not find the IUV or INDS file!')
            continue
        

        ke=np.unique(INDS[:,:])
        #ke中非零的值即为检测到的人的索引值
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`the keys in INDS:',ke)


        # personI = np.zeros([im.shape[0],im.shape[1]])
        

        if ke.size<1:
            print('can not find any person in this image!')
            continue
        contourKP_inter=im.copy()
        contourKP=im.copy()
        myKPtemp=im.copy()


        size_IMG=im.shape[1]*im.shape[0]

        # INDS_inter=copy.deepcopy(INDS[:,:,0])
        # INDS_interA=INDS_inter.reshape(1,-1)
        # INDS_interB=INDS_interA[0].tolist()
        # size_personK=INDS_interB.count(2)
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',size_personK)





        # ageProto = "./models/age_gender_models/age_deploy.prototxt"
        # ageModel = "./models/age_gender_models/age_net.caffemodel"

        # genderProto = "./models/age_gender_models/gender_deploy.prototxt"
        # genderModel = "./models/age_gender_models/gender_net.caffemodel"

        # MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        # ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        # genderList = ['Male', 'Female']

        # ageNet = cv2.dnn.readNetFromCaffe(ageProto,ageModel )
        # genderNet = cv2.dnn.readNetFromCaffe(genderProto,genderModel)








        All_persons=np.zeros([im.shape[0],im.shape[1]], dtype=np.uint8)
        All_persons[INDS[:,:]!=0]=255
        cv2.imwrite(args.output_dir+os.path.basename(im_name).split('.')[0]+'_INDS_bi.png',All_persons)
        
        KPoint_file='DensePoseData/infer_out/KPoints.txt'
        f1 = open(KPoint_file, 'a')
        f1.writelines(['File:',os.path.basename(im_name),',',str(im.shape[1]),'X',str(im.shape[0]),'\n'])
        
        OUT_Mask_path=args.output_dir+os.path.basename(im_name).split('.')[0]
        
        

        for i in range(ke.size-1):
            # print('re_personID:',ke[i+1]) #从第一个人开始处理
            # personI[INDS==ke[i]]=i+1
            aPerson=np.zeros([im.shape[0],im.shape[1]], dtype=np.uint8)
            aPerson[INDS[:,:]==ke[i+1]]=255
            OUT_Mask_DIR=OUT_Mask_path+'/'+'INDS_person_'+str(ke[i+1])+'.png'
            cv2.imwrite(OUT_Mask_DIR,aPerson)

            INDS_inter=copy.deepcopy(INDS[:,:])
            INDS_interA=INDS_inter.reshape(1,-1)
            INDS_interB=INDS_interA[0].tolist()
            size_personK=INDS_interB.count(ke[i+1])
            
            # print('the size of person #:',size_personK)
            # print('the percent:',size_personK/(size_IMG*1.0))

            if size_personK/(size_IMG*1.0)<0.01:
                print('the size of this person is too small!!') #不足图片1/100大小的人不处理
                continue

            
            
            IUV_inter=copy.deepcopy(IUV)
            IUV_inter[INDS!=ke[i+1]]=0
            
            
            
            # #添加性别、年龄检测，男性或低于12岁的儿童不处理
            # #以下代码识别性别、年龄不准确，暂不添加此功能
            # faceMask = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
            # # x,y=np.where((IUV_inter[:,:,0]==23)|(IUV_inter[:,:,0]==24)) #左右半部分头部
            # # faceMask[x,y]=IUV_inter[x,y]
            # faceMask[(IUV_inter[:,:,0]==23)|(IUV_inter[:,:,0]==24)]=IUV_inter[(IUV_inter[:,:,0]==23)|(IUV_inter[:,:,0]==24)]
            # personFace=faceMask[:,:,0]

            # _, personFace_binary = cv2.threshold(personFace,0,255,cv2.THRESH_BINARY) 
            # _, contours, _ = cv2.findContours(personFace_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # if len(contours)<1:
            #     print('cannot find this face')
            #     continue
            # face_x,face_y,face_w,face_h = cv2.boundingRect(contours[0])
            # # cv2.rectangle(contourKP_inter,(face_x,min(face_y+5,im.shape[0])),(face_x+face_w,min(face_y+5+int(face_h*0.88),im.shape[0])),(0,255,0),2)
            

            # face=contourKP_inter[min(face_y+5,im.shape[0]):min(face_y+5+int(face_h*0.88),im.shape[0]),face_x:face_x+face_w]
            
            # cv2.imwrite(args.output_dir+os.path.basename(im_name).split('.')[0]+'_face'+str(ke[i+1])+'.png',face)

 
            # faceI = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)  

            # blob = cv2.dnn.blobFromImage(faceI, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # genderNet.setInput(blob)
            # genderPreds = genderNet.forward()
            # gender = genderList[genderPreds[0].argmax()]
            # if gender==genderList[0]:
            #     print("Gender : {}, conf = {:.3f}, without processing".format(gender, genderPreds[0].max()))
            #     continue
            # # print("Gender Output : {}".format(genderPreds))
            # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            # ageNet.setInput(blob)
            # agePreds = ageNet.forward()
            # age = ageList[agePreds[0].argmax()]
            # if age==ageList[0] or age==ageList[1] or age==ageList[2]:
            #     print("Age : {}, conf = {:.3f}, without processing".format(age, agePreds[0].max()))
            #     continue
            # print("Age Output : {}".format(agePreds))
            # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            # label = "{},{}".format(gender, age)
            # cv2.putText(contourKP_inter, label, (face_x,face_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.imwrite(args.output_dir+os.path.basename(im_name).split('.')[0]+'_face.png',contourKP_inter)








            f1.writelines(['+++++++++++++++++++\n'])
            f1.writelines(['# of person:', str(i+1),'\n'])

            # PartIdarray=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)
            procI=0 #默认处理所有身体部分
            PartIdarray=pre_process_PartID(procI)

            iTempKP_name=args.output_dir+os.path.basename(im_name).split('.')[0]
            
            myKPtemp=findContourKP(myKPtemp,iTempKP_name,aPerson,All_persons, IUV_inter,keypoints0[i], PartIdarray,f1)
            # myKPtemp=findOutcontourKP(myKPtemp,iTempKP_name,All_persons, IUV_inter,keypoints0[i], PartIdarray,f1)

            contourKP=processPart(contourKP_inter,All_persons, IUV_inter, PartIdarray,f1)  #默认处理所有检测到的符合条件的人
            contourKP_inter=contourKP

            # cv2.imwrite('DensePoseData/infer_out/val_results/'+os.path.basename(im_name).split('.')[0]+'_KP_'+str(i+1)+'.png',contourKP)

            
            # contourKP=processPart(contourKP_inter, IUV, PartIdarray,INDS[:,:],ke[i+1])  #默认处理所有人
            f1.writelines(['\n\n'])
            
        f1.writelines(['-------------------------------\n\n\n'])
        f1.flush()      #将修改写入到文件中（无需关闭文件）
        cv2.imwrite(args.output_dir+os.path.basename(im_name).split('.')[0]+'_KP.png',contourKP)

    
    f1.close()



if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)

