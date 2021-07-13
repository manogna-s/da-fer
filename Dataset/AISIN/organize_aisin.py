import numpy as np
import pandas as pd
import shutil

exp2label = {'Happy': 0, 'Neutral': 1}
label2exp = {0: 'Happy', 1: 'Neutral'}
split = 'test'
img_labels = np.array(pd.read_csv(f'./lists/image_list_{split}.txt', header = None, delim_whitespace=True))

n_imgs = img_labels.shape[0]
print(n_imgs)
train_annotations=[]
test_annotations=[]
for i in range(n_imgs):
    img_file = img_labels[i][0]
    label=img_labels[i][1]
    expression = label2exp[label]
    if split=='train':
        source = f'/home/manogna/Dataset/AISIN/new_images/'+img_file
        dest = './Train/'+expression+'/'+img_file
    elif split=='test':
        source = f'/home/manogna/Dataset/AISIN/new_images/'+img_file
        dest = './Test/'+expression+'/'+img_file
    
    shutil.copy(source, dest)