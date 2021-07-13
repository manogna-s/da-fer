import numpy as np
import pandas as pd
import shutil

exp2label = {'Surprised': 0, 'Fear': 1, 'Disgust': 2, 'Happy': 3, 'Sad': 4, 'Anger': 5, 'Neutral': 6}
label2exp = {1:'Surprised', 2:'Fear', 3:'Disgust', 4:'Happy', 5:'Sad', 6:'Anger', 7:'Neutral'}

img_labels = np.array(pd.read_csv('./list_patition_label.txt', header = None, delim_whitespace=True))

n_imgs = img_labels.shape[0]

train_annotations=[]
test_annotations=[]
for i in range(n_imgs):
    img_file = img_labels[i][0]
    label=img_labels[i][1]
    source = './basic/Image/original/'+img_file
    expression = label2exp[label]
    if img_file[:5]=='train':
        dest = './Train/'+expression+'/'+img_file
    elif img_file[:4]=='test':
        dest = './Test/'+expression+'/'+img_file
    
    shutil.copy(source, dest)
    
