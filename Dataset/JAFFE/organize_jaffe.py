import numpy as np
import pandas as pd
import shutil

exp2label = {'Surprised': 0, 'Fear': 1, 'Disgust': 2, 'Happy': 3, 'Sad': 4, 'Anger': 5, 'Neutral': 6}
label2exp = {0:'Surprised', 1:'Fear', 2:'Disgust', 3:'Happy', 4:'Sad', 5:'Anger', 6:'Neutral'}

img_labels = np.array(pd.read_csv('./list_putao.txt', header = None, delim_whitespace=True))

n_imgs = img_labels.shape[0]

train_annotations=[]
test_annotations=[]
for i in range(n_imgs):
    img_file = img_labels[i][0]
    label=img_labels[i][1]
    source = './images/'+img_file
    expression = label2exp[label]
    dest = './Train/'+expression+'/'+img_file
    shutil.copy(source, dest)
    
