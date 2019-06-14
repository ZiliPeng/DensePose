# -*- coding: utf-8 -*-
import numpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def contourPointIDs(countourL,c_point,offset):
    if c_point+offset>=countourL:
        pointId=countourL-1
    elif c_point+offset<0:
        pointId=0
    else:
        pointId=c_point+offset

    return pointId

#画最外边轮廓的关键点
def drawContourKP(im,contours,cX,cY,outer=True,PartId=2,maxAreaId=0):

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
    if ((LeftBody==True) and (RightBody==False)):
        cv2.circle(im, (contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1]), 2, (255, 0, 0), -1) #左边关键点
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,KPstep)
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,-KPstep)
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
    if ((RightBody==True) and (LeftBody==False)):
        cv2.circle(im, (contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1]), 2, (255, 0, 0), -1) #右边关键点
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,KPstep)
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,-KPstep)
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
    if ((LeftBody==True) and (RightBody==True)):
        cv2.circle(im, (contours[maxAreaId][HMminIdx,0,0], contours[maxAreaId][HMminIdx,0,1]), 2, (255, 0, 0), -1)
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,-int(KPstep*0.75))
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMminIdx,int(KPstep*1.1))
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1) #左边腰部
        cv2.circle(im, (contours[maxAreaId][HMmaxIdx,0,0], contours[maxAreaId][HMmaxIdx,0,1]), 2, (255, 0, 0), -1)
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,int(KPstep*0.75))
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1)
        offsetId=contourPointIDs(contours[maxAreaId][:,0,0].size,HMmaxIdx,-int(KPstep*1.1))
        cv2.circle(im, (contours[maxAreaId][offsetId,0,0], contours[maxAreaId][offsetId,0,1]), 2, (255, 0, 0), -1) #右边腰部

    return Short2Center




def processPart(im, IUV, PartIdarray):
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





        # PartId=PartIdarray[1]
        # for PartId in PartIdarray:
        # img = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)

        for index in range(len(PartIdarray)):
            rePartId=PartIdarray[index]
            print(rePartId)
     

            # fig = plt.gcf()
            fig = plt.figure(figsize=(xinch,yinch))
            # fig.set_size_inches(xinch,yinch)
            plt.axis('off') 

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)

            canvas2 = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
            plt.imshow( canvas2[:,:,::-1] )


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
                print('无法识别的身体ID代号！')
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
            



            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11, 11))
            
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
                continue
            

            maxArea=0
            maxAreaId=0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i])>maxArea:
                    maxArea=cv2.contourArea(contours[i])
                    maxAreaId=i

            print('面积最大的轮廓',maxAreaId)
            # xxx=cv2.drawContours(im, contours, maxAreaId,(0,255,0), 1, cv2.LINE_AA)

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
            Short2Center=drawContourKP(im,contours,cX,cY,True,rePartId,maxAreaId=maxAreaId)




            
            
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
                continue
            

            maxArea=0
            maxAreaId=0
            for i in range(len(contours)):
                if cv2.contourArea(contours[i])>maxArea:
                    maxArea=cv2.contourArea(contours[i])
                    maxAreaId=i
            xxx=cv2.drawContours(im, contours, maxAreaId,(0,0,255), 1, cv2.LINE_AA)


            # 重新计算腐蚀后的重心点坐标compute the center of the contour
            M = cv2.moments(contours[maxAreaId])
            if M["m00"]==0:
                M["m00"]=0.001
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            drawContourKP(im,contours,cX,cY,False,rePartId,maxAreaId=maxAreaId)


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



        cv2.imwrite('../DensePoseData/infer_out/slim/000288_contour.png',xxx)



def main():

    im  = cv2.imread('../DensePoseData/demo_data/test_jpg/000288.jpg')
    IUV = cv2.imread('../DensePoseData/infer_out/ziliInfer/000288_IUV.png')
    INDS = cv2.imread('../DensePoseData/infer_out/ziliInfer/000288_INDS.png')

    w_ratio = 0.25
    h_ratio = 0.25
    #obtain the original data size
    im_size = im.shape
    #set the size of the object date
    size = (int(im_size[1] * w_ratio),int(im_size[0] * h_ratio)) 
    #resize the original data
    im_resized =cv2.resize(im,size,interpolation = cv2.INTER_CUBIC)

    # cv2.imshow('原图', im_resized)
    # cv2.waitKey()

    ke=np.unique(INDS[:,:,0])
    print(ke)       #ke中非零的即为检测到的人的索引值


    
    # PartIdarray=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24)
    procI=1


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

    
    processPart(im, IUV, PartIdarray)






if __name__ == "__main__":
    main()