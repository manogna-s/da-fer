import os
original_file = open(os.path.join("/home/megh/projects/fer/da-fer/Dataset/RAF/lists/list_patition_label.txt"))
file_ = open("image_list.txt","w")
#file_test = open("Ne_Hp_class_list_test.txt","w")

for line in original_file:
    line = line.replace("\n","")
    if int(line.split()[1]) == 4:
        file_.write(line.split()[0] + " " + "0")
        file_.write("\n") # condition for happy/smiling expression
    if int(line.split()[1]) == 7:
        file_.write(line.split()[0] + " " + "1")
        file_.write("\n") # condition for neutral expression
