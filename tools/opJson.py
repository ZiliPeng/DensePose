# -*- coding: utf-8 -*-
import numpy as np
import json
import PIL.Image
import os
import cv2

def read_json(json_path,thresh=0.7):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    
    person=0
    partIdBase=0
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)   #第n个人的KP
        person=person+1
        # print('-------------------person:',person)
        # print(len(kp))
        partIdBase=len(kp)*(person-1)
        for i in range(len(kp)):
            partIdtmp=i

            if kp[i,2]>thresh and partIdtmp!=0 and partIdtmp!=15 and partIdtmp!=16 and partIdtmp!=17 and partIdtmp!=18:
                kp[i,2]=partIdBase+partIdtmp
                # print(partIdBase+partIdtmp)
                # print(kp[i,:])
                kps.append(kp[i,:].tolist())
    return kps


def get_bbox(json_path, vis_thr=0.2):
    kps = read_json(json_path)
    # Pick the most confident detection.
    scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    kp = kps[np.argmax(scores)]
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        import ipdb
        ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    return scale, center



img_name="DensePoseData/demo_data/test_jpg/000288.jpg"
json_path="DensePoseData/demo_data/test_jpg/000288_keypoints.json"  #OpenPose的输出结果
outputDir="DensePoseData/infer_out/ziliInfer/"



kps = read_json(json_path,0.6)


src_img = np.array(PIL.Image.open(img_name))
img = src_img.copy()

print(kps)
kps=np.array(kps)

print(kps.shape)
persons=int(kps[-1,2]/24)+1
print(persons)



for i in range(kps.shape[0]):
    # print(kps[i,:2])
    cv2.circle(img, (int(kps[i,0]),int(kps[i,1])), 2, [0,255,0], -1, cv2.LINE_AA)
# cv2.imshow("Keypoints",img[:,:,::-1])
# cv2.waitKey(0)
cv2.imwrite(outputDir + os.path.basename(img_name).split('.')[0]+'_poseKP.jpg',img[:,:,::-1])


# PIL.Image.fromarray(skel_rend.astype(np.uint8)).save(outputDir + os.path.basename(img_name).split('.')[0]+'_skel.png')

