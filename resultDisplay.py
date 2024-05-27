
import cv2
from glob import glob
import matplotlib.pyplot as plt
import os,re


dir_name = "./result/png"
files = glob("./result/png/*.png")
print(files)

input_files = [file for file in files if 'input' in file ]

for  input_file in input_files:
    print(input_file)
    base_name = os.path.basename(input_file)
    print(base_name)

    plt.figure(figsize=(10, 8))
    plt.subplot(1,3,1)
    fname = os.path.join(dir_name , base_name)
    img = cv2.imread(fname)
    plt.imshow(img)

    plt.subplot(1,3,2)
    label_name = re.sub(r"input", "label", base_name)
    fname = os.path.join(dir_name , label_name)
    img = cv2.imread(fname)
    plt.imshow(img)

    plt.subplot(1,3,3)
    output_name = re.sub(r"input", "output", base_name)
    fname = os.path.join(dir_name , output_name)
    img = cv2.imread(fname)

    plt.imshow(img)
    plt.show()







