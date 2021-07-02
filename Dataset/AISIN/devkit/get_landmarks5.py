# Author: Tao Pu
# Date: 2019.09.14

import os
import cv2
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN

landmark_path = os.path.join("/home/megh/projects/fer/da-fer/Dataset/AISIN/landmarks_5")
if not os.path.exists(landmark_path):
    os.mkdir(landmark_path)

detector = MTCNN()

list_patition_label = pd.read_csv('/home/megh/projects/fer/da-fer/Dataset/AISIN/lists/image_list.txt', header=None, delim_whitespace=True)
list_patition_label = np.array(list_patition_label)

for index in range(list_patition_label.shape[0]):
    imgPath = os.path.join('/home/megh/projects/fer/da-fer/Dataset/AISIN/images', list_patition_label[index,0].replace("train_","").replace("test_",""))
    print(imgPath)
    image_original = cv2.imread(imgPath)
    
    #img = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    img = image_original
    
    preds = detector.detect_faces(img)

    if preds is None or len(preds)==0:
            continue

    pred = preds[0]
    landmarks = []

    landmark = pred['keypoints']
    landmarks.append(landmark['right_eye'])
    landmarks.append(landmark['left_eye'])
    landmarks.append(landmark['nose'])
    landmarks.append(landmark['mouth_right'])
    landmarks.append(landmark['mouth_left'])
    print(index)
    np.savetxt(os.path.join('./',list_patition_label[index,0][:-3]+'txt'),np.array(landmarks).astype(np.int))
    
