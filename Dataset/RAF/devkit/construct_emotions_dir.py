import os
import shutil
import random

SrcPath='../basic/EmoLabel/list_patition_label.txt'
EmotionDir='../basic/Emotions'
ListDir='../list_experiment'

f=open(SrcPath, 'r')
lines = f.readlines()

path = os.path.join(ListDir, 'train_id.txt')
f_train = open(path, 'w')
path = os.path.join(ListDir, 'val_id.txt')
f_val = open(path, 'w')

random.shuffle(lines)

for line in lines:
    im_name, emotion = line.split()
    id, ext = im_name.split('.')
    phase, idx = id.split('_')

    path = os.path.join(EmotionDir, id+'.txt')
    f_emotion = open(path, 'w')
    f_emotion.write(emotion)
    f_emotion.write('\n')
    f_emotion.close()

    out_line = id + '\n'
    if phase == 'train':
        f_train.write(out_line)
    else:
        f_val.write(out_line)


