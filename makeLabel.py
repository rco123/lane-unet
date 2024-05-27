import os
import cv2
from glob import glob
import numpy as np
import math, random


def split_number_with_ratio(total_count, ratio1, ratio2, ratio3):
    # Calculate the total ratio
    total_ratio = ratio1 + ratio2 + ratio3
    
    # Calculate the base parts using integer division
    part1 = (ratio1 * total_count) // total_ratio
    part2 = (ratio2 * total_count) // total_ratio
    part3 = (ratio3 * total_count) // total_ratio
    
    # Calculate the remainder to ensure the sum is exactly total_count
    remainder = total_count - (part1 + part2 + part3)
    
    # Distribute the remainder to ensure the sum is equal to total_count
    parts = [part1, part2, part3]
    
    # Distribute the remainder starting from the largest ratio part
    ratios = [ratio1, ratio2, ratio3]
    for i in range(remainder):
        max_ratio_index = ratios.index(max(ratios))
        parts[max_ratio_index] += 1
        ratios[max_ratio_index] = 0  # Prevent adding again to the same part
    
    return tuple(parts)


def angle_to_endpoint(spoint, angle):

    dline = 150
    rangle = angle * math.pi / 180

    dx = dline * -math.sin(rangle)
    dy = dline  * math.cos(rangle)

    epoint = (spoint[0] + int(dx), spoint[1] - int(dy))

    print(epoint)
    return epoint

def fname_to_angle(file):
    file = os.path.basename(file)
    angle = int(file.split('_')[-1].split('.')[0])
    angle = int(angle / 20)
    return angle

def get_key():
    while True:
        if cv2.waitKey(1) == 32 :
            break
        if cv2.waitKey(1) == ord('q'):
            exit(1)


files = glob('**/*.jpg', recursive=True)
print(len(files))

random.shuffle(files)

ntrain, nval, ntest = split_number_with_ratio(len(files), 8, 1, 1)

print(ntrain, nval, ntest)

dir_train = "./dataset/train"
dir_val = "./dataset/val"
dir_test = "./dataset/test"

if not os.path.exists(dir_train):
    os.makedirs(dir_train)
if not os.path.exists(dir_val):
    os.makedirs(dir_val)
if not os.path.exists(dir_test):
    os.makedirs(dir_test)

def make_img(dname,file):

    img = cv2.imread(file)
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("win", gimg)
    print("gimg shape = ", gimg.shape)
    ## make mask

    height , width = gimg.shape[0], gimg.shape[1]
    mask = np.zeros(gimg.shape, dtype=np.uint8)
    
    spoint = ( int(width/2), height)

    angle = fname_to_angle(file)
    epoint =  angle_to_endpoint(spoint, angle)

    #epoint = ( int(width/2), 100)
    color = 255
    thickness = 20
    cv2.line(mask, spoint, epoint, color, thickness    )
    cv2.imshow("mask", mask)

    # combined_image = cv2.bitwise_or(gimg, mask)
    # cv2.imshow("combined", combined_image)

    new_size = (128,128)
    
    base_name = os.path.basename(file)
    fname = os.path.join(dname, base_name)
    rsimg = cv2.resize(gimg, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(fname, rsimg)

    base_name = os.path.basename(file)
    fname = os.path.join(dname, "label_" + base_name)
    rsimg = cv2.resize(mask, new_size, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(fname, rsimg)

for cnt in range(ntrain):
    file = files[cnt]
    print(file)
    make_img(dir_train, file)

for cnt in range(nval):
    file = files[cnt + ntest]
    print(file)
    make_img(dir_val, file)

for cnt in range(ntest):
    file = files[cnt + ntest + nval]
    print(file)
    make_img(dir_test, file)




