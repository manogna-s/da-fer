import os
import numpy as np

annos_dir= '../basic/Annotation/boundingbox'
annos = os.listdir(annos_dir)


SizeCount=np.zeros(4, dtype=np.int32)
Thre = 56
for anno in annos:
    fin = open(os.path.join(annos_dir, anno))
    line = fin.readline()
    strs = line.split(' ')

    bboxstrs = strs[0:4]
    bbox = np.zeros(4, dtype=np.int32)
    for idx in range(4):
        #print(idx)
        #print(bboxstrs[idx])
        bbox[idx] = int(float(bboxstrs[idx]))

    #bbox=[int(bboxstrs[0]), int(bboxstrs[1]), int(bboxstrs[2]), int(bboxstrs[3])]
    bbox_w = bbox[2]-bbox[0]
    bbox_h = bbox[3]-bbox[1]

    if min(bbox_w, bbox_h) < 28:
        SizeCount[0] = SizeCount[0]+1
    else:
        if min(bbox_w, bbox_h) < 56:
            SizeCount[1] = SizeCount[1]+1
        else:
           if min(bbox_w, bbox_h) < 112:
               SizeCount[2] = SizeCount[2]+1
           else:
               SizeCount[3] = SizeCount[3]+1

print(SizeCount)
