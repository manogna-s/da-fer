import os
import shutil

from PIL.Image import new
from matplotlib.pyplot import new_figure_manager

image_path = os.path.join("/home/megh/projects/fer/da-fer/Dataset/AISIN/images")
image_list = open(os.path.join("/home/megh/projects/fer/da-fer/Dataset/AISIN/lists/image_list.txt"))
new_image_path = os.path.join("/home/megh/projects/fer/da-fer/Dataset/AISIN/new_img")
for image_name in image_list:
    image_name = str(image_name).replace("\n","").split()[0]
    if "train" in str(image_name):
        indicator = "train_"
    elif "test" in str(image_name):
        indicator = "test_"
    actual_name = str(image_name).replace("train_","").replace("test_","")
    old_name = os.path.join(image_path,actual_name)
    new_name = os.path.join(new_image_path,indicator + actual_name)
    shutil.copyfile(old_name,new_name) 
    print("copied")
