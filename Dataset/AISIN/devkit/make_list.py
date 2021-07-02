import os
import random
import sys
dataset_path = os.path.join("/home/megh/projects/fer/da-fer/Dataset/AISIN/images")

file_ = open(os.path.join("./image_list.txt"),"w")
#file_test = open(os.path.join("./image_list_test.txt"),"w")

for idx, emo in enumerate(os.listdir(os.path.join(dataset_path))):
    print(emo)
    emo_path = os.path.join(dataset_path,str(emo))
    for image in os.listdir(os.path.join(emo_path)):
        if random.random() > 0.78:
            file_.write("test_" + str(image) + " " + str(idx))
            file_.write("\n")
        else:
            file_.write("train_" + str(image) + " " + str(idx))
            file_.write("\n")





