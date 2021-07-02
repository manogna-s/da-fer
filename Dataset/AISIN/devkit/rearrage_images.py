import os
import shutil

old_path = os.path.join("/home/megh/projects/fer/da-fer/Dataset/AISIN/images_original")
new_path = os.path.join("/home/megh/projects/fer/da-fer/Dataset/AISIN/images/")

for face_id in os.listdir(old_path):
    face_path = os.path.join(old_path,str(face_id))
    for emotion in os.listdir(face_path):
        image_path = os.path.join(face_path,str(emotion))
        for image in os.listdir(image_path):
            path_from_copy = os.path.join(image_path,image)
            path_to_copy = os.path.join(new_path,str(emotion))
            shutil.copy(path_from_copy,path_to_copy)
            print("copied")