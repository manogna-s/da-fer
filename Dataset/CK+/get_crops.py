import os
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import json
import pandas as pd

dataset = 'CK+'
detector = MTCNN()

exp2label = {'Surprised': 0, 'Fear': 1, 'Disgust': 2, 'Happy': 3, 'Sad': 4, 'Anger': 5, 'Neutral': 6}

for split in ['test', 'train']:
    if split == 'train':
        img_folder = 'Train/'
    if split == 'test':
        img_folder = 'Test/'

    annotations = {split: []}
    for expression in os.listdir(img_folder):
        for img in os.listdir(img_folder + expression):
            img_path = os.path.join(img_folder, expression, img)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            preds = detector.detect_faces(img)

            if preds is None or len(preds) == 0:
                continue

            pred = preds[0]
            landmarks = pred['keypoints']

            landmark = []
            landmark.append(landmarks['right_eye'])
            landmark.append(landmarks['left_eye'])
            landmark.append(landmarks['nose'])
            landmark.append(landmarks['mouth_right'])
            landmark.append(landmarks['mouth_left'])

            box = pred['box']
            x1 = box[0]
            y1 = box[1]
            x2 = x1 + box[2]
            y2 = y1 + box[3]
            bbox = [x1, y1, x2, y2]

            annot = {'img_path': '../Dataset/'+dataset+'/'+img_path, 'expression': expression, 'label': exp2label[expression]}
            annot['bbox'] = bbox
            annot['landmarks'] = landmark
            annotations[split].append(annot)
            print(annot)

    with open(f'{split}_annotations.json', 'w') as fp:
        json.dump(annotations, fp)

    df = pd.DataFrame.from_dict(annotations[split])
    df.to_csv(f'{split}.csv', index=False, header=True)
