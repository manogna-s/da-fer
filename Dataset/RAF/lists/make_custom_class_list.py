import os
original_file = open(os.path.join("list_patition_label.txt"))
file_ = open("image_list.txt","w")
train_file = open('image_list_train.txt', 'w')
test_file = open('image_list_test.txt', 'w')

for line in original_file:
    line = line.replace("\n","")
    if int(line.split()[1]) == 4:
        file_.write(line.split()[0] + " " + "0")
        file_.write("\n") # condition for happy/smiling expression
        if line.split()[0][:5] == 'train':
            train_file.write(line.split()[0] + " " + "0\n")
        if line.split()[0][:4] == 'test':
            test_file.write(line.split()[0] + " " + "0\n")
    if int(line.split()[1]) == 7:
        file_.write(line.split()[0] + " " + "1")
        file_.write("\n") # condition for neutral expression
        if line.split()[0][:5] == 'train':
            train_file.write(line.split()[0] + " " + "1\n")
        if line.split()[0][:4] == 'test':
            test_file.write(line.split()[0] + " " + "1\n")
