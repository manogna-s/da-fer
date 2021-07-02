import os
import numpy as np

LABEL=[2,4,3,5,6,1,0]

InDir = '../basic/Emotions_ori'
OutDir = '../basic/Emotions'

files=os.listdir(InDir)
print(len(files))

for file in files:
  path = os.path.join(InDir, file)
  idx = int(np.loadtxt(path)) - 1
  
  path = os.path.join(OutDir, file)
  f = open(path, 'w')
  line = '{}\n'.format(LABEL[idx])
  f.write(line)
  print(idx, line)
  f.close()
